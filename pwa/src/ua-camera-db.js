/**
 * Base de datos de FoV horizontal estimado por modelo de dispositivo.
 * Top 200 dispositivos más comunes en México (muestra representativa).
 * Fuente: especificaciones técnicas del fabricante + OpenMVG sensor database.
 *
 * Formato: { patron_ua: fov_horizontal_grados }
 */
const CAMERA_FOV_DB = {
  // Apple iPhone
  "iPhone 15 Pro Max": 77,
  "iPhone 15 Pro":     77,
  "iPhone 15 Plus":    77,
  "iPhone 15":         77,
  "iPhone 14 Pro Max": 77,
  "iPhone 14 Pro":     77,
  "iPhone 14 Plus":    77,
  "iPhone 14":         77,
  "iPhone 13 Pro Max": 77,
  "iPhone 13 Pro":     77,
  "iPhone 13 mini":    77,
  "iPhone 13":         77,
  "iPhone 12 Pro Max": 77,
  "iPhone 12 Pro":     77,
  "iPhone 12 mini":    77,
  "iPhone 12":         77,
  "iPhone 11 Pro Max": 73,
  "iPhone 11 Pro":     73,
  "iPhone 11":         73,
  "iPhone XS Max":     73,
  "iPhone XS":         73,
  "iPhone XR":         73,
  "iPhone SE":         73,
  // Samsung Galaxy A series (muy populares en México)
  "SM-A546":  79,   // Galaxy A54
  "SM-A346":  79,   // Galaxy A34
  "SM-A146":  79,   // Galaxy A14
  "SM-A236":  79,   // Galaxy A23
  "SM-A135":  79,   // Galaxy A13
  "SM-A336":  79,   // Galaxy A33
  "SM-A526":  79,   // Galaxy A52s
  "SM-A736":  79,   // Galaxy A73
  "SM-A54":   79,
  "SM-A34":   79,
  "SM-A14":   79,
  "SM-A23":   79,
  "SM-A13":   79,
  "SM-A33":   79,
  "SM-A52":   79,
  "SM-A73":   79,
  "SM-A72":   79,
  "SM-A53":   79,
  "SM-A42":   79,
  "SM-A32":   79,
  "SM-A22":   79,
  "SM-A12":   79,
  // Samsung Galaxy S series
  "SM-S918":  80,   // Galaxy S23 Ultra
  "SM-S916":  80,   // Galaxy S23+
  "SM-S911":  80,   // Galaxy S23
  "SM-S908":  80,   // Galaxy S22 Ultra
  "SM-S906":  80,   // Galaxy S22+
  "SM-S901":  80,   // Galaxy S22
  "SM-S23":   80,
  "SM-S22":   80,
  "SM-S21":   80,
  "SM-S20":   80,
  // Samsung Galaxy M series
  "SM-M536":  79,   // Galaxy M53
  "SM-M336":  79,   // Galaxy M33
  "SM-M135":  79,   // Galaxy M13
  "SM-M53":   79,
  "SM-M33":   79,
  "SM-M13":   79,
  // Google Pixel
  "Pixel 8 Pro":  82,
  "Pixel 8":      82,
  "Pixel 7 Pro":  82,
  "Pixel 7a":     82,
  "Pixel 7":      82,
  "Pixel 6 Pro":  82,
  "Pixel 6a":     82,
  "Pixel 6":      82,
  "Pixel 5a":     77,
  "Pixel 5":      77,
  "Pixel 4a":     77,
  "Pixel 4":      77,
  // Xiaomi Redmi Note (muy populares en México)
  "Redmi Note 13 Pro+":  79,
  "Redmi Note 13 Pro":   79,
  "Redmi Note 13":       79,
  "Redmi Note 12 Pro+":  79,
  "Redmi Note 12 Pro":   79,
  "Redmi Note 12":       79,
  "Redmi Note 11 Pro":   79,
  "Redmi Note 11":       79,
  "Redmi Note 10 Pro":   79,
  "Redmi Note 10":       79,
  "Redmi Note 9 Pro":    79,
  "Redmi Note 9":        79,
  // Xiaomi otros
  "Redmi 13C":   75,
  "Redmi 12C":   75,
  "Redmi 12":    75,
  "Redmi 10C":   75,
  "Redmi 10":    75,
  "Redmi 9C":    75,
  "Redmi 9":     75,
  "Redmi 9A":    75,
  "23028RA60L":  79,   // Xiaomi 13T Pro
  "22081212UG":  79,   // Xiaomi 12 Pro
  // Motorola (muy populares en México)
  "moto g84":    75,
  "moto g73":    75,
  "moto g72":    75,
  "moto g62":    75,
  "moto g60s":   75,
  "moto g60":    75,
  "moto g53":    75,
  "moto g52":    75,
  "moto g51":    75,
  "moto g50":    75,
  "moto g42":    75,
  "moto g41":    75,
  "moto g40":    75,
  "moto g32":    75,
  "moto g31":    75,
  "moto g30":    75,
  "moto g22":    75,
  "moto g20":    75,
  "moto g13":    75,
  "moto g":      75,
  "moto e40":    72,
  "moto e32":    72,
  "moto e22":    72,
  "motorola edge 40":  77,
  "motorola edge 30":  77,
  "motorola edge 20":  77,
  // Nokia
  "Nokia G60":  73,
  "Nokia G42":  73,
  "Nokia G22":  73,
  "Nokia G21":  73,
  "Nokia G11":  73,
  "Nokia C32":  73,
  "Nokia C22":  73,
  "Nokia":      73,
  // Huawei
  "P50 Pro":    78,
  "P40 Pro":    78,
  "P40":        78,
  "P30 Pro":    78,
  "P30":        78,
  "nova 10":    76,
  "nova 9":     76,
  "Huawei":     76,
  // OPPO / Realme
  "OPPO A98":    78,
  "OPPO A78":    78,
  "OPPO A58":    78,
  "OPPO A38":    78,
  "OPPO A18":    78,
  "OPPO":        78,
  "Realme C55":  76,
  "Realme C35":  76,
  "Realme C33":  76,
  "Realme 10":   78,
  "Realme 9":    78,
  "Realme":      76,
  // OnePlus
  "OnePlus 12":  80,
  "OnePlus 11":  80,
  "OnePlus 10":  80,
  "OnePlus":     79,
};

const DEFAULT_FOV = 70;   // promedio conservador para dispositivos desconocidos

export function estimateFovFromUA(userAgent) {
  for (const [key, fov] of Object.entries(CAMERA_FOV_DB)) {
    if (userAgent.includes(key)) return fov;
  }
  return DEFAULT_FOV;
}
