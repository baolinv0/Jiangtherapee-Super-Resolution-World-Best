"""Recover four fusion WGSL shaders from the digest-pinned official JSR v9.8 EXE.

Uses Python's standard library only. Reads PE resources and the embedded
PyInstaller archive as data; never imports or executes any released code.
Usage: python extract_released_shaders.py JSR_V9_8.exe --output evidence
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib

RELEASE_URL = 'https://github.com/y-g-jiang/Jiangtherapee-Super-Resolution-World-Best/releases/download/v9.8/JSR_V9_8.exe'
RELEASE_BYTES = 318034141
RELEASE_SHA256 = 'e4fbe9a44ba508e368a7802bb5b28be1e8a49fc666d28f8306f41d34cf3576c3'
MODULE_SHA256 = '6724796493551c05ef53f8b40196fcd55924c78963bdc24bd9d401b90d344150'
COOKIE_MAGIC = b'MEI\014\013\012\013\016'
COOKIE_FORMAT = '!8sIIII64s'
COOKIE_BYTES = struct.calcsize(COOKIE_FORMAT)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_at(stream, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0:
        raise ValueError('Negative file extent')
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f'Truncated input at {offset}, expected {size} bytes')
    return data


def resource_records(stream) -> list[dict]:
    if read_at(stream, 0, 2) != b'MZ':
        raise ValueError('Expected a PE executable')
    pe = struct.unpack('<I', read_at(stream, 60, 4))[0]
    if read_at(stream, pe, 4) != b'PE\0\0':
        raise ValueError('Invalid PE signature')
    header = struct.unpack('<HHIIIHH', read_at(stream, pe + 4, 20))
    section_count, optional_size = header[1], header[5]
    optional = pe + 24
    magic = struct.unpack('<H', read_at(stream, optional, 2))[0]
    directory = optional + {0x20b: 112, 0x10b: 96}[magic]
    resource_rva, resource_size = struct.unpack('<II', read_at(stream, directory + 16, 8))
    sections = []
    for i in range(section_count):
        section = struct.unpack('<8sIIIIIIHHI', read_at(stream, optional + optional_size + i * 40, 40))
        sections.append((section[2], section[1], section[4], section[3]))

    def file_offset(rva: int, size: int = 1) -> int:
        for virtual, virtual_size, raw, raw_size in sections:
            if virtual <= rva and rva + size <= virtual + min(virtual_size, raw_size):
                return raw + rva - virtual
        raise ValueError('Resource points outside backed PE sections')

    root = file_offset(resource_rva)
    records = []
    active = set()

    def parse(relative: int, names: tuple = ()) -> None:
        if relative in active or len(names) > 8 or relative + 16 > resource_size:
            raise ValueError('Invalid PE resource tree')
        active.add(relative)
        directory_header = struct.unpack('<IIHHHH', read_at(stream, root + relative, 16))
        count = directory_header[-2] + directory_header[-1]
        if relative + 16 + count * 8 > resource_size:
            raise ValueError('Resource directory exceeds section')
        for i in range(count):
            key, value = struct.unpack('<II', read_at(stream, root + relative + 16 + i * 8, 8))
            if key & 0x80000000:
                name_offset = key & 0x7fffffff
                length = struct.unpack('<H', read_at(stream, root + name_offset, 2))[0]
                name = read_at(stream, root + name_offset + 2, length * 2).decode('utf-16-le')
            else:
                name = key
            if value & 0x80000000:
                parse(value & 0x7fffffff, names + (name,))
            else:
                rva, size, codepage, reserved = struct.unpack('<IIII', read_at(stream, root + value, 16))
                records.append({'resource': list(names + (name,)), 'offset': file_offset(rva, size), 'bytes': size})
        active.remove(relative)

    parse(0)
    return records


def archive_table(stream, resource: dict) -> tuple[int, dict, dict]:
    tail_size = min(65536, resource['bytes'])
    tail_offset = resource['offset'] + resource['bytes'] - tail_size
    tail = read_at(stream, tail_offset, tail_size)
    position = tail.rfind(COOKIE_MAGIC)
    if position < 0 or position + COOKIE_BYTES > len(tail):
        raise ValueError('Embedded PyInstaller archive footer not found')
    magic, archive_size, table_offset, table_size, python_version, library = struct.unpack(
        COOKIE_FORMAT, tail[position:position + COOKIE_BYTES])
    archive_end = tail_offset + position + COOKIE_BYTES
    archive_start = archive_end - archive_size
    if archive_start < resource['offset'] or table_offset + table_size > archive_size - COOKIE_BYTES:
        raise ValueError('Invalid embedded archive extent')
    data = read_at(stream, archive_start + table_offset, table_size)
    table = {}
    cursor = 0
    while cursor < len(data):
        if cursor + 18 > len(data):
            raise ValueError('Truncated archive table record')
        length, offset, compressed, expanded, compression, kind = struct.unpack_from('!IIIIBc', data, cursor)
        if length < 18 or cursor + length > len(data):
            raise ValueError('Invalid archive table record size')
        name = data[cursor + 18:cursor + length].rstrip(b'\0').decode('utf-8')
        if offset + compressed > table_offset or compression not in (0, 1):
            raise ValueError('Archive record points outside data area')
        table[name] = (offset, compressed, expanded, compression, kind.decode('ascii'))
        cursor += length
    info = {'offset': archive_start, 'bytes': archive_size, 'python_version': python_version,
            'python_library': library.rstrip(b'\0').decode('ascii'), 'table_offset': archive_start + table_offset,
            'table_bytes': table_size}
    return archive_start, table, info


def shader_strings(module: bytes) -> dict[str, bytes]:
    results = {}
    for match in re.finditer(rb'[\x09\x0a\x0d\x20-\x7e]{80,}', module):
        text = match.group()
        if b'@compute' not in text or b'struct Params' not in text:
            continue
        # Nuitka serializes this Unicode constant with a leading 'u' marker.
        # Canonical text starts at the WGSL struct and uses LF with one final LF.
        text = text[text.index(b'struct Params'):].replace(b'\r\n', b'\n').strip() + b'\n'
        if b'scale_counts' in text:
            name = 'phase_finalize'
        elif b'qmap' in text and b'TWO_PI' in text:
            name = 'learned_accumulate'
        elif b'legacy_num' in text and b'array<f32, 16>' in text:
            name = 'phase_accumulate'
        elif b'select(0.0' in text and b'den > 0.0' in text:
            name = 'learned_finalize'
        else:
            raise ValueError('Unrecognized embedded compute shader')
        if name in results:
            raise ValueError(f'Duplicate embedded shader: {name}')
        results[name] = text
    expected = {'phase_accumulate', 'phase_finalize', 'learned_accumulate', 'learned_finalize'}
    if set(results) != expected:
        raise ValueError(f'Expected four shaders, found {sorted(results)}')
    return results


def extract(executable: Path, output: Path) -> dict:
    if executable.stat().st_size != RELEASE_BYTES:
        raise ValueError(f'Expected {RELEASE_BYTES} bytes for the official release')
    digest = hashlib.sha256()
    with executable.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        if digest.hexdigest() != RELEASE_SHA256:
            raise ValueError('Official release SHA-256 mismatch')
        resources = resource_records(stream)
        resource = next(record for record in resources if record['resource'] == [10, 101, 1033])
        if read_at(stream, resource['offset'], 2) != b'MZ':
            raise ValueError('Backend resource is not a PE executable')
        start, table, archive = archive_table(stream, resource)
        module_name = next(name for name in table if name.replace('\\', '/').split('/')[-1] == 'wgpu_multik_stream_v1.cp310-win_amd64.pyd')
        offset, compressed_size, size, compression, kind = table[module_name]
        packed = read_at(stream, start + offset, compressed_size)
        module = zlib.decompress(packed) if compression else packed
        if len(module) != size or sha256(module) != MODULE_SHA256:
            raise ValueError('Released streaming fusion extension digest mismatch')
    shaders = shader_strings(module)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data in sorted(shaders.items()):
        filename = name + '.wgsl'
        (output / filename).write_bytes(data)
        files[name] = {'file': filename, 'bytes': len(data), 'sha256': sha256(data)}
    manifest = {'release': 'v9.8', 'release_url': RELEASE_URL, 'executable_bytes': RELEASE_BYTES,
                'executable_sha256': RELEASE_SHA256, 'static_only': True, 'backend_resource': resource,
                'archive': archive, 'module': {'name': module_name,
                'archive_offset': start + offset, 'compressed_bytes': compressed_size, 'bytes': size,
                'compressed_sha256': sha256(packed), 'sha256': sha256(module)},
                'text_normalization': 'Start at struct Params, LF newlines, strip surrounding whitespace, append one LF',
                'shaders': files}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = extract(args.executable, args.output)
    print(json.dumps({'verified_release': manifest['executable_sha256'], 'shaders': manifest['shaders'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
