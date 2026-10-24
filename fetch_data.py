"""Download and verify the exact public environmental snapshot used in the pilot."""
from pathlib import Path
import json,hashlib,subprocess
R=Path(__file__).resolve().parent
manifest=json.loads((R/'data/source_manifest.json').read_text())
for name,info in manifest['files'].items():
 p=R/'data'/Path(name).name
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:
  subprocess.run(['curl','-fsSL','--retry','2','--max-time','120',info['url'],'-o',str(p)],check=True)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:raise RuntimeError(f'Checksum mismatch: {name}')
wurl='https://archive-api.open-meteo.com/v1/archive?latitude=43.338996&longitude=76.903239&start_date=2021-12-30&end_date=2025-12-31&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation,boundary_layer_height&models=era5&timezone=GMT&wind_speed_unit=ms'
p=R/'data/weather_era5.json'
if not p.exists():subprocess.run(['curl','-fsSL','--retry','2','--max-time','120',wurl,'-o',str(p)],check=True)
print('AirData snapshot verified:',manifest['commit'])
print('Weather requested from ERA5 via Open-Meteo:',wurl)
print('Weather SHA256:',hashlib.sha256(p.read_bytes()).hexdigest())
