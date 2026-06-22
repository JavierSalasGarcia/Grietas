"""Initial schema — tabla reports con PostGIS

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import geoalchemy2

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Archivos
        sa.Column("image_path", sa.Text, nullable=False),
        sa.Column("metadata_json", postgresql.JSONB, nullable=False),
        # Clasificación
        sa.Column("categoria", sa.String(20)),
        sa.Column("confianza", sa.Float),
        sa.Column("processing_status", sa.String(20), server_default="queued"),
        sa.Column("error_message", sa.Text),
        # Geolocalización
        sa.Column(
            "location",
            geoalchemy2.types.Geometry("POINT", srid=4326, nullable=True),
        ),
        # Métricas de grieta
        sa.Column("crack_width_mm", sa.Float),
        sa.Column("crack_length_mm", sa.Float),
        sa.Column("crack_area_mm2", sa.Float),
        sa.Column("crack_fractal", sa.Float),
        sa.Column("crack_angle_deg", sa.Float),
        sa.Column("crack_area_ratio", sa.Float),
        sa.Column("crack_branches", sa.Integer),
        sa.Column("severity_class", sa.String(10)),
        sa.Column("gsd_mm_px", sa.Float),
        # InSAR
        sa.Column("subsid_mm", sa.Float),
        sa.Column("subsid_vel_mm_yr", sa.Float),
        sa.Column("subsid_accel", sa.Float),
        # Riesgo
        sa.Column("risk_index", sa.Float),
        # Serie temporal
        sa.Column(
            "previous_report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reports.id"),
            nullable=True,
        ),
        sa.Column("delta_width_mm", sa.Float),
        sa.Column("delta_length_mm", sa.Float),
        sa.Column("growth_rate_mm_mo", sa.Float),
        # Calibración
        sa.Column("intrinsics_source", sa.String(30)),
        sa.Column("ortho_quality", sa.String(10)),
    )

    op.create_index(
        "ix_reports_location",
        "reports",
        ["location"],
        postgresql_using="gist",
    )
    op.create_index("ix_reports_created_at", "reports", ["created_at"])
    op.create_index("ix_reports_processing_status", "reports", ["processing_status"])


def downgrade() -> None:
    op.drop_index("ix_reports_processing_status", table_name="reports")
    op.drop_index("ix_reports_created_at", table_name="reports")
    op.drop_index("ix_reports_location", table_name="reports")
    op.drop_table("reports")
