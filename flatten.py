from pathlib import Path
import shutil
script_dir = Path(__file__).parent.resolve()

for subfolder in list(script_dir.iterdir()):
    if subfolder.is_dir():
        for file_path in list(subfolder.rglob('*')):
            if file_path.is_file():
                destination_path = script_dir / file_path.name
                if destination_path.exists():
                    destination_path = script_dir / f"{file_path.stem}_copy{file_path.suffix}"

                shutil.move(str(file_path), str(destination_path))
        shutil.rmtree(subfolder)
print("All files brought to the main folder and empty subfolders removed!")