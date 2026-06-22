"""
Tareas Celery para procesamiento asíncrono de reportes.
Cada reporte clasificado como Aprobado pasa por: ortorectificación → segmentación.
"""
import asyncio

from celery import Celery

from api.config import REDIS_URL

celery_app = Celery("grietas", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_report(self, report_id: str):
    """
    Pipeline completo para un reporte Aprobado:
    1. Clasificar (si aún no se hizo en tiempo real)
    2. Ortorectificar la imagen oblicua
    3. Segmentar grieta y extraer métricas
    4. Consultar subsidencia InSAR
    5. Calcular índice de riesgo
    6. Actualizar registro en DB
    """
    asyncio.run(_process_report_async(self, report_id))


async def _process_report_async(task, report_id: str):
    from api.database import AsyncSessionLocal
    from api.models import Report

    async with AsyncSessionLocal() as db:
        report = await db.get(Report, report_id)
        if not report:
            return

        try:
            # Paso 1 — Clasificación
            report.processing_status = "classifying"
            await db.commit()

            from api.classifier import classifier

            if classifier is None:
                raise RuntimeError("Modelo ONNX no disponible")

            result = classifier.predict(report.image_path)
            report.categoria = result["categoria"]
            report.confianza = result["confianza"]

            if result["categoria"] == "No Aprobado":
                report.processing_status = "done"
                await db.commit()
                return

            # Paso 2 — Ortorectificación
            report.processing_status = "orthorectifying"
            await db.commit()

            import cv2

            from api.orthorectify import orthorectify

            image = cv2.imread(report.image_path)
            meta = report.metadata_json
            ortho, gsd, quality = orthorectify(image, meta)
            ortho_path = report.image_path.replace(".jpg", "_ortho.jpg")
            cv2.imwrite(ortho_path, ortho)
            report.ortho_quality = quality

            # Paso 3 — Segmentación y métricas
            report.processing_status = "segmenting"
            await db.commit()

            from api.crack_analysis import analyze_crack

            metrics = analyze_crack(ortho, gsd)
            report.crack_width_mm = metrics["crack_width_mm"]
            report.crack_length_mm = metrics["crack_length_mm"]
            report.crack_area_mm2 = metrics["crack_area_mm2"]
            report.crack_fractal = metrics["crack_fractal_dim"]
            report.crack_angle_deg = metrics["crack_angle_deg"]
            report.crack_area_ratio = metrics["crack_area_ratio"]
            report.crack_branches = metrics["crack_branches"]
            report.severity_class = metrics["severity_class"]
            report.gsd_mm_px = metrics.get("gsd_mm_px", gsd)
            report.intrinsics_source = (
                meta.get("camera", {}).get("intrinsics_source")
            )

            # Paso 4 — Subsidencia InSAR
            if report.location is not None:
                try:
                    from geoalchemy2.shape import to_shape

                    from api.insar import get_subsidence_at_point

                    point = to_shape(report.location)
                    subsid = await get_subsidence_at_point(point.x, point.y, db)
                    if subsid:
                        report.subsid_mm = subsid["displacement_mm"][-1]
                        report.subsid_vel_mm_yr = subsid["velocity_mm_yr"]
                        report.subsid_accel = subsid["acceleration"]
                except Exception:
                    pass  # InSAR opcional; no bloquea el pipeline

            # Paso 5 — Índice de riesgo
            from api.risk import compute_risk_index

            report.risk_index = compute_risk_index(report)

            report.processing_status = "done"
            await db.commit()

        except Exception as exc:
            report.processing_status = "error"
            report.error_message = str(exc)
            await db.commit()
            raise task.retry(exc=exc)
