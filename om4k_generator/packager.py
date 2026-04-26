import zipfile
import io
from typing import Dict, Tuple

class Packager:
    @staticmethod
    def package(audio_bytes: bytes, audio_filename: str, bg_bytes: bytes, bg_filename: str, osu_files: Dict[str, str]) -> bytes:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr(audio_filename, audio_bytes)
            if bg_bytes and bg_filename:
                zip_file.writestr(bg_filename, bg_bytes)
                
            for filename, content in osu_files.items():
                zip_file.writestr(filename, content)
                
        return zip_buffer.getvalue()
