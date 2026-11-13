#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Umbra Denoise (BETA), Copyright (c) 2026 Kishan Patel. Own code; no code taken from other scripts.
"""AI denoise for linear images. Apply shows the result in the window; Apply to Siril replaces the image (Undo available).
Denoise: strength of the noise removal. Detail: gives back part of the removed fine detail. Pattern noise: removes blotchy mottle repeated in every sub.
Run in Siril (Scripts menu), or standalone:
  python Umbra_Denoise.py --image in.fits --out out.fits [--model PATH.onnx] [--denoise 0..1] [--detail 0..1] [--pattern 0..1]
  python Umbra_Denoise.py --ui-test        (opens the window on a fake image, runs Apply once, exits)
  python Umbra_Denoise.py --check-updates | --download [MODEL_ID|all]"""
VERSION = '0.4.0'
# 0.4.0: edits carry over between tools (Denoise, then Deblur starts from the denoised image); Undo / Redo above the image (Cmd/Ctrl+Z); Undo all fixed (back to the opened image, also in Siril); Apply is a small blue button inside the panel; drop-down menus show the whole list; hold Space anywhere to see the original, [ and ] change the brush size; second panel with Restore (paint back the image from before the tool), Clone (drag the blue source ring, paint to copy, like Umbra Astro Studio) and Eraser brushes.
# 0.3.0: new window like Umbra Guided GHS: fixed controls column, zoomable image on the right that shows the original at once; Apply shows the result there, Apply to Siril writes it, Undo all puts back the starting image; Download sits next to the model list, models not installed are listed as (Not Downloaded), and the update check runs at every start (green / yellow note).
# 0.2.0: GPU by default also on Linux with NVIDIA (CUDA libraries installed on first run if missing); OpenVINO / MIGraphX used when installed; half-size model files with identical results.
# 0.1.1: buttons, check boxes, drop-down menus and scroll bars now look the same as in Umbra Astro Studio.
# 0.1.0: First family release (ONNX Runtime, no PyTorch): Preview window, Cancel, model download and update check, remembered settings.
# ============================================================================================ shared code (src/common.py)
# Everything the five scripts share: dependency install, ONNX Runtime sessions, updater, preferences, Siril I/O, family UI, CLI.
import sys, os, json, hashlib, argparse, time, math, re, types
from pathlib import Path
_STANDALONE = any(a in ('--image', '--ui-test', '--download', '--check-updates') or a.startswith(('--image=', '--download=')) for a in sys.argv)
if __name__ == '__main__' and not _STANDALONE and '--help' not in sys.argv and '-h' not in sys.argv:
    import sirilpy as _siril_deps
    def _ort_requirement():
        """onnxruntime variant for this machine (all variants import as 'onnxruntime'; only one may be installed). An existing one is kept."""
        import importlib.metadata as _md, platform as _pf, shutil as _sh
        for n in ('onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-rocm', 'onnxruntime-migraphx', 'onnxruntime-openvino', 'onnxruntime'):
            try:
                _md.version(n); return n
            except Exception:
                pass
        if _pf.system() == 'Windows': return 'onnxruntime-directml'          # GPU through DirectX 12 (NVIDIA / AMD / Intel), no extra downloads
        if _pf.system() == 'Linux' and _sh.which('nvidia-smi'): return 'onnxruntime-gpu'   # CUDA; the CUDA 12 / cuDNN 9 libraries are installed below if missing
        return 'onnxruntime'                                                   # macOS: Core ML is built in; Linux without NVIDIA: CPU
    _ORT_PKG = _ort_requirement()
    _siril_deps.ensure_installed('numpy', _ORT_PKG, 'PyQt6', 'scipy', 'astropy')
    if _ORT_PKG == 'onnxruntime-gpu':
        # GPU by default on NVIDIA: if the system has no CUDA 12 / cuDNN 9 libraries, install NVIDIA's pip packages of them
        # (onnxruntime loads them with preload_dlls()).  Large one-time download (about 2-3 GB); CPU fallback if anything fails.
        import ctypes.util as _cu
        if not _cu.find_library('cudnn'):
            try:
                _siril_deps.ensure_installed('nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu12', 'nvidia-cudnn-cu12', 'nvidia-cufft-cu12',
                                             'nvidia-curand-cu12', 'nvidia-cuda-nvrtc-cu12')
            except Exception as _e:
                print(f'Could not install the CUDA libraries ({_e}); the script will use the CPU.', flush=True)
import numpy as np
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINES = {}      # tool key -> engine namespace (filled by the build)
TOOLS = {}        # tool key -> ToolSpec (filled by the build)
USER_AGENT = 'UmbraAstro'    # generic client name only; nothing about the user or the machine is sent
_REDIRECT_HOSTS = {'github.com', 'objects.githubusercontent.com', 'release-assets.githubusercontent.com', 'raw.githubusercontent.com'}
GITHUB_USER = 'Kishanhpatel'


class ToolSpec:
    """Everything that differs between the tools (the shared code reads only this)."""
    def __init__(self, **kw):
        self.legacy_repos = ()
        self.fallback_folder = None
        self.labels = {}
        self.default_stems = ()
        self.notes = []
        self.linear_only = True
        self.preview_whole = True        # Apply runs on the whole image and shows the result in the pane
        self.reuse_preview = True        # Apply to Siril with unchanged settings reuses the result shown
        self.alt = None                  # (button text, info key) second preview layer
        self.progress_unit = 'tile'
        self.extras = {}                 # extra model files: kind -> file stem prefix (e.g. {'critic': 'umbra_starless_critic'}); not shown in the model picker
        self.has_editor = False          # Starless: an Edit marks button opens the correction-marks editor
        self.inline_preview = True       # every tool: Apply shows the result in the pane, Apply to Siril writes it
        self.restore_brush = True        # second card with the Restore brush (Starless has its own Remove / Keep card instead)
        self.apply_top = False           # Starless: Apply sits on the left right under the model row; the panel puts its controls in panel.card2
        self.__dict__.update(kw)

    @property
    def title(self):
        return f'{self.name} (BETA)'

    # hooks (overridden by the tool modules)
    def build_panel(self, ui, layout, host):
        raise NotImplementedError
    def add_cli(self, ap):
        pass
    def params_from_args(self, a):
        return {}
    def image_params(self, header_text):
        return {}
    def preview_params(self, params):
        return params
    def label(self, params):
        return ''
    def output(self, params, out, info):
        return out
    def post_apply(self, ctx, params, info):
        return ''
    def check_cli(self, a, ap):
        pass
    def post_cli(self, a, dst, info, header, label):
        return ''
    def prepare_job(self, view, job):
        """GUI thread, before a run starts: whatever the run needs from the window (snapshot).  Handed to engine_run as `extra`."""
        return None
    def engine_run(self, eng, data, params, progress, cancel, extra):
        return eng.run(data, params, progress=progress, cancel=cancel)


def env(spec, name, default=None):
    return os.environ.get(spec.prefix + name, default)


def offline_mode(spec):
    return bool(env(spec, 'OFFLINE') or os.environ.get('UMBRA_OFFLINE'))


# ----------------------------------------------------------------------------------------------- models folder
def _onnx_files(d):
    try:
        return sorted(Path(d).glob('*.onnx')) if Path(d).is_dir() else []
    except OSError:
        return []


def model_dir(spec):
    """(folder, used_fallback). <PREFIX>MODEL_DIR wins; else '<folder>' next to the script; Deblur falls back to the old
    'Umbra Deconvolve' folder when its own is missing or empty and the old one has models."""
    e = env(spec, 'MODEL_DIR')
    if e:
        return Path(e), False
    main = SCRIPT_DIR / spec.folder
    if spec.fallback_folder and not _onnx_files(main) and _onnx_files(SCRIPT_DIR / spec.fallback_folder):
        return SCRIPT_DIR / spec.fallback_folder, True
    return main, False


def download_dir(spec):
    e = env(spec, 'MODEL_DIR')
    return Path(e) if e else SCRIPT_DIR / spec.folder


_MANIFEST_NAMES = {}      # (tool, file stem) -> name from the manifest (after an update check)


def model_label(spec, stem):
    return spec.labels.get(stem) or _MANIFEST_NAMES.get((spec.key, stem)) or stem


def is_extra(spec, name):
    """True for a file of an extra kind (e.g. the Auto-keep critic): it is a model file, but never offered as a model."""
    stem = Path(str(name)).stem
    return any(stem.startswith(pre) for pre in spec.extras.values())


def extra_path(spec, kind):
    """Newest installed file of an extra kind (e.g. 'critic') in the models folder, or None.  <PREFIX><KIND>_MODEL overrides."""
    e = env(spec, kind.upper() + '_MODEL')
    if e:
        return e if Path(e).is_file() else None
    pre = spec.extras.get(kind)
    if not pre:
        return None
    files = [f for f in _onnx_files(download_dir(spec)) if f.stem.startswith(pre)]
    if not files:
        files = [f for f in _onnx_files(model_dir(spec)[0]) if f.stem.startswith(pre)]
    return str(sorted(files, key=lambda f: (_version_tuple(f.stem.rsplit('_v', 1)[-1].replace('_experimental', '')) or (0,), f.name))[-1]) if files else None


def list_models(spec):
    items = [(model_label(spec, f.stem), str(f)) for f in _onnx_files(model_dir(spec)[0]) if not is_extra(spec, f)]
    items.sort(key=lambda it: (0 if Path(it[1]).stem in spec.default_stems else 1, it[0]))
    extra = getattr(spec, 'extra_models', None)              # e.g. Starless: StarNet++ when the user has it installed
    if callable(extra):
        try:
            items += list(extra())
        except Exception:
            pass
    return items


def default_model_path(spec, saved=None):
    e = env(spec, 'MODEL')
    if e:
        return e
    d = model_dir(spec)[0]
    if saved:
        p = d / Path(saved).name
        if p.suffix == '.onnx' and p.is_file():
            return str(p)
    items = list_models(spec)
    return items[0][1] if items else None


# ----------------------------------------------------------------------------------------------- preferences
class Prefs:
    """Umbra_<Tool>_prefs.json: {"model": <file name>, "settings": {...}}; combined: {"tool": key, "tools": {key: {...}}}."""
    def __init__(self, path, combined=False, enabled=True):
        self.path, self.combined, self.enabled = Path(path), combined, enabled

    def _read(self):
        try:
            d = json.loads(self.path.read_text())
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def load(self, key):
        d = self._read()
        if self.combined:
            d = (d.get('tools') or {}).get(key) or {}
        s = d.get('settings')
        return (d.get('model') if isinstance(d.get('model'), str) else None), (s if isinstance(s, dict) else {})

    def load_tool(self):
        t = self._read().get('tool')
        return t if isinstance(t, str) else None

    def save(self, key, model, settings, current=None):
        if not self.enabled:
            return
        try:
            entry = {'model': Path(model).name if model else None, 'settings': settings}
            if self.combined:
                d = self._read()
                tools = d.get('tools') if isinstance(d.get('tools'), dict) else {}
                tools[key] = entry
                out = {'tool': current or key, 'tools': tools}
            else:
                out = entry
            self.path.write_text(json.dumps(out, indent=1))
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------- ONNX Runtime
_IMPORT_ERROR = {}   # first onnxruntime / scipy import failure (remembered so every later attempt shows the real cause)


def log_file(spec):
    return SCRIPT_DIR / f'{spec.script}_error.log'


def write_error_log(spec, text):
    try:
        log_file(spec).write_text(f'{time.ctime()}\npython {sys.executable}\n\n{text}')
    except Exception:
        pass


def import_engine(spec):
    """Import onnxruntime and scipy exactly once; on failure keep the ORIGINAL error (with traceback) for every later attempt."""
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR['msg'])
    try:   # Siril's working folder may have been deleted; some imports then fail in os.getcwd() -> move to a folder that exists
        os.getcwd()
    except OSError:
        os.chdir(str(SCRIPT_DIR))
    try:
        import onnxruntime  # noqa: F401
        import scipy.ndimage  # noqa: F401
    except BaseException as e:
        import traceback
        tb = traceback.format_exc()
        _IMPORT_ERROR['msg'] = f'{type(e).__name__}: {e} (full details: {log_file(spec)})'
        write_error_log(spec, f'sys.path {sys.path}\n\n{tb}')
        print(tb, flush=True)
        raise RuntimeError(_IMPORT_ERROR['msg']) from e


def make_session_factory(prefix, skip=()):
    """-> factory(model_path, feed) -> (run_fn, provider_label, note).  run_fn(feed_dict) -> list of outputs (like sess.run(None, feed)).
    GPU providers are tried in order of preference for the platform (CUDA, DirectML, ROCm, MIGraphX, OpenVINO, Core ML); each is checked against the CPU
    provider on one probe feed (outputs must agree to 1e-3 relative) and the first that works is used. Anything missing, failing or
    disagreeing falls through to the next, and finally to the CPU.  <PREFIX>CPU=1 forces the CPU."""
    def factory(model_path, feed, dims=None):
        import onnxruntime as ort
        path = str(model_path)
        def new_opts():
            o = ort.SessionOptions(); o.log_severity_level = 3
            for k, v in (dims or {}).items():          # pin free dimensions (Core ML compiles a fixed shape fast, a free one very slowly)
                o.add_free_dimension_override_by_name(k, int(v))
            return o
        so = new_opts()
        holder = []
        def mk_cpu():
            if not holder:
                holder.append(ort.InferenceSession(path, so, providers=['CPUExecutionProvider']))
            return holder[0]
        note = None
        sess, label = None, 'CPU'
        if not os.environ.get(prefix + 'CPU'):
            avail = ort.get_available_providers()
            if 'CUDAExecutionProvider' in avail and hasattr(ort, 'preload_dlls'):
                try:
                    ort.preload_dlls()          # CUDA / cuDNN from NVIDIA's pip packages, when the system has none
                except Exception:
                    pass
            cands = [('CUDAExecutionProvider', 'CUDA', {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC'}, False),
                     ('DmlExecutionProvider', 'DirectML', {}, True),
                     ('ROCMExecutionProvider', 'ROCm', {}, False),
                     ('MIGraphXExecutionProvider', 'MIGraphX', {}, False),
                     ('OpenVINOExecutionProvider', 'OpenVINO', {'device_type': 'GPU'}, False),
                     ('CoreMLExecutionProvider', 'Core ML', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}, False)]
            notes = []
            for name, lab, opts, dml in cands:
                if name not in avail or lab in skip:
                    continue
                try:
                    so2 = new_opts()
                    if dml:                                    # DirectML requirements
                        so2.enable_mem_pattern = False; so2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    g = ort.InferenceSession(path, so2, providers=[(name, opts), 'CPUExecutionProvider'])
                    if name not in g.get_providers():
                        raise RuntimeError('provider did not load')
                    a, b = g.run(None, feed), mk_cpu().run(None, feed)
                    err = max(float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-9)) for x, y in zip(a, b))
                    if np.isfinite(err) and err < 1e-3:
                        sess, label = g, lab
                        break
                    notes.append(f'{lab} result differed from the CPU ({err:.1e})')
                except Exception as e:
                    notes.append(f'{lab} unavailable ({str(e).strip().splitlines()[0][:90] if str(e).strip() else type(e).__name__})')
            if sess is None and notes:
                note = '; '.join(notes) + '; using the CPU.'
            elif sess is None:
                note = 'No GPU provider in this onnxruntime build; using the CPU.'
        if sess is None:
            sess = mk_cpu()
        return (lambda f: sess.run(None, f)), label, note
    return factory


_ENGINE_CACHE = {}     # tool key -> (model path, Engine)   (one model in memory per tool; the combined window unloads the others)


def get_engine(spec, model_path):
    c = _ENGINE_CACHE.get(spec.key)
    if c and c[0] == str(model_path):
        return c[1]
    _ENGINE_CACHE.pop(spec.key, None)
    import_engine(spec)
    p = Path(str(model_path or ''))
    if not p.is_file():
        raise FileNotFoundError(f'Model not found: {p}')
    eng = ENGINES[spec.key].Engine(str(p), make_session_factory(spec.prefix))
    if callable(getattr(spec, 'configure_engine', None)):
        spec.configure_engine(eng)
    _ENGINE_CACHE[spec.key] = (str(p), eng)
    return eng


def unload_engines(keep=None):
    for k in list(_ENGINE_CACHE):
        if k != keep:
            _ENGINE_CACHE.pop(k, None)
    import gc; gc.collect()


def stretched_warning(data):
    """Message if the data looks stretched (median > 0.2 of max), else None."""
    d = np.asarray(data, np.float32)
    fin = d[np.isfinite(d)]
    if fin.size == 0:
        return None
    mx = float(fin.max())
    if mx > 0 and float(np.median(fin)) > 0.2 * mx:
        return 'the data looks stretched (median above 20% of the maximum); this tool expects linear data. Running anyway.'
    return None


# ----------------------------------------------------------------------------------------------- model download / updates
class DownloadError(Exception):
    pass


class UpdSpec:
    """Where a script looks for models: its manifest repo and every repo whose release assets / release page it accepts."""
    def __init__(self, repo, allowed=None):
        self.repo = repo
        self.allowed = list(allowed or [repo])
        self.manifest_url = f'https://raw.githubusercontent.com/{repo}/main/models/models.json'
        self.releases_url = f'https://github.com/{repo}/releases'


def updspec_for(spec):
    return UpdSpec(spec.repo, [spec.repo, *spec.legacy_repos])


def combined_upd(specs):
    """The combined script's own manifest repo; it accepts model / script URLs from the repos of all its tools (no re-upload)."""
    me = f'{GITHUB_USER}/umbra-astro-ai'
    return UpdSpec(me, [me] + [t.repo for t in specs] + [r for t in specs for r in t.legacy_repos])


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        from urllib.parse import urlparse
        u = urlparse(newurl)
        if u.scheme != 'https' or u.hostname not in _REDIRECT_HOSTS:
            raise DownloadError(f'Refused a redirect to {u.hostname or newurl}')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _ssl_context():
    """Default certificate store; if the Python build has none (python.org macOS builds without 'Install Certificates'), use certifi's.
    Certificates are always verified."""
    import ssl
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get('x509_ca') and not ctx.cert_store_stats().get('x509'):
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
    return ctx


def _open(url, timeout=20):
    """The only network entry point: HTTPS GET with a generic user agent; redirects only to GitHub hosts; certificates verified."""
    opener = urllib.request.build_opener(_SafeRedirect(), urllib.request.HTTPSHandler(context=_ssl_context()))
    return opener.open(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=timeout)


def _valid_model_url(url, file, upd):
    from urllib.parse import urlparse
    u = urlparse(str(url))
    return (u.scheme == 'https' and u.hostname == 'github.com' and not u.query and u.path.endswith('/' + file)
            and any(u.path.startswith(f'/{r}/releases/download/') for r in upd.allowed))


def _valid_entry(m, upd):
    try:
        f = m['file']
        return (re.fullmatch(r'[A-Za-z0-9._-]{1,120}\.onnx', f) is not None and '..' not in f and _valid_model_url(m['url'], f, upd)
                and re.fullmatch(r'[0-9a-f]{64}', str(m['sha256']).lower()) is not None and int(m['size_bytes']) > 0 and bool(m.get('name'))
                and re.fullmatch(r'[a-z]{1,20}', str(m.get('kind', 'model'))) is not None)
    except Exception:
        return False


def _version_tuple(v):
    try:
        return tuple(int(p) for p in str(v).strip().lstrip('v').split('.'))
    except Exception:
        return None


def parse_manifest(data, upd):
    """Validated manifest dict {'models': [...valid entries...], 'script': {'version','url'} or None}. Bad entries and entries marked
    "legacy" (kept only for old script versions) are dropped."""
    if not isinstance(data, dict) or data.get('schema') != 1 or not isinstance(data.get('models'), list):
        raise DownloadError('The model list has an unexpected format.')
    models = [dict(m, sha256=str(m['sha256']).lower(), size_bytes=int(m['size_bytes'])) for m in data['models']
              if isinstance(m, dict) and not m.get('legacy') and _valid_entry(m, upd)]
    sc = data.get('script')
    if not (isinstance(sc, dict) and _version_tuple(sc.get('version')) and any(str(sc.get('url', '')).startswith(f'https://github.com/{r}/') for r in upd.allowed)):
        sc = None
    return dict(models=models, script=sc)


_MANIFEST_CACHE = {}


def fetch_manifest(upd, url=None, use_cache=False):
    key = url or upd.manifest_url
    c = _MANIFEST_CACHE.get(key)
    if use_cache and c and time.time() - c[0] < 300:
        return c[1]
    try:
        with _open(key, 20) as resp:
            raw = resp.read(1_000_000)
        man = parse_manifest(json.loads(raw.decode('utf-8')), upd)
        _MANIFEST_CACHE[key] = (time.time(), man)
        return man
    except DownloadError:
        raise
    except Exception as e:
        why = ' Certificate problem: on macOS run "Install Certificates.command" for your Python.' if 'CERTIFICATE' in str(e).upper() else ''
        raise DownloadError(f'Could not reach the model list ({type(e).__name__}). Offline? The installed models still work.{why}') from e


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def model_state(entry, folder):
    """'missing' | 'current' | 'update' for a manifest entry against the installed file."""
    p = Path(folder) / entry['file']
    if not p.is_file():
        return 'missing'
    # alt_sha256: older files that give identical results (e.g. the same weights stored at full size), so they are not re-downloaded
    ok = {entry['sha256']} | {str(h).lower() for h in (entry.get('alt_sha256') or []) if isinstance(h, str)}
    return 'current' if file_sha256(p) in ok else 'update'


def entries_for(manifest, tool=None):
    """Manifest entries for one tool (entries with a 'tool' field belong to that tool; without one they belong to every single script)."""
    return [m for m in manifest['models'] if tool is None or m.get('tool') in (None, tool)]


def check_updates(manifest, spec, folder=None):
    """-> dict(models=[(entry, state)], script_version, script_url, script_newer)."""
    folder = folder or download_dir(spec)
    sc = manifest.get('script')
    newer = bool(sc) and _version_tuple(sc['version']) > (_version_tuple(VERSION) or (0,))
    for m in manifest['models']:
        _MANIFEST_NAMES[(m.get('tool') or spec.key, Path(m['file']).stem)] = m['name']
    return dict(models=[(m, model_state(m, folder)) for m in entries_for(manifest, spec.key)],
                script_version=sc['version'] if sc else None, script_url=sc['url'] if sc else None, script_newer=newer)


def download_model(entry, upd, folder, progress=None, cancel=None):
    """Download one manifest entry into the models folder: temp file, size and SHA-256 check, then moved into place.  Returns the path."""
    if not _valid_entry(entry, upd):
        raise DownloadError('Refused: not a valid model entry.')
    d = Path(folder)
    try:
        d.mkdir(parents=True, exist_ok=True)
        free = __import__('shutil').disk_usage(d).free
    except OSError as e:
        raise DownloadError(f'Cannot write to the models folder {d}: {e}') from e
    size = int(entry['size_bytes'])
    if free < size * 1.1:
        raise DownloadError(f'Not enough free disk space in {d} ({size / 1e6:.0f} MB needed).')
    part, final = d / f'.{entry["file"]}.part', d / entry['file']
    h = hashlib.sha256(); got = 0
    try:
        with _open(entry['url'], 30) as resp, open(part, 'wb') as out:
            while True:
                if cancel and cancel():
                    raise DownloadError('Download cancelled.')
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
                if got > size * 1.01 + 1024:
                    raise DownloadError('The download is larger than expected; stopped.')
                out.write(chunk); h.update(chunk)
                if progress: progress(got, size)
        if got != size:
            raise DownloadError(f'Incomplete download ({got} of {size} bytes).')
        if h.hexdigest() != entry['sha256']:
            raise DownloadError('Checksum mismatch: the file was not installed. Try again later.')
        os.replace(part, final)
        return final
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f'Download failed ({type(e).__name__}: {e}).') from e
    finally:
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass


def describe_updates(res, spec):
    """Plain-text summary lines for a check_updates() result."""
    lines = []
    if res['script_newer']:
        lines.append(f"A newer script is available: {res['script_version']} (you have {VERSION}). {res['script_url']}")
    elif res['script_version']:
        lines.append(f'The script is up to date ({VERSION}).')
    for m, st in res['models']:
        if st == 'missing':
            lines.append(f"Available to download: {m['name']} ({m['size_bytes'] / 1e6:.0f} MB).")
        elif st == 'update':
            lines.append(f"Update available for {m['name']} (the installed file differs from the published one).")
    if not any(st != 'current' for _, st in res['models']):
        lines.append('Installed models are up to date.')
    return lines


def missing_model_message(spec, res=None):
    """Status text when no model is installed."""
    if res:
        d = next((m for m, st in res['models'] if st == 'missing' and m.get('default') and not m.get('kind')), None) or next((m for m, st in res['models'] if st == 'missing' and not m.get('kind')), None)
        if d:
            return f"No model installed. Select {d['name']} in the dropdown and click Download ({d['size_bytes'] / 1e6:.0f} MB)."
    return f'No model installed. Select a model in the dropdown and click Download, or put a .onnx file in {model_dir(spec)[0]}.'


# ----------------------------------------------------------------------------------------------- Siril I/O
def read_pixels(siril):
    data = siril.get_image_pixeldata()
    if data is None:
        fit = siril.get_image(with_pixels=True)
        data = fit.data if fit is not None else None
    if data is None:
        raise RuntimeError('No image loaded in Siril.')
    return np.array(data, copy=True)


def decode_pixels(raw):
    """Siril pixels -> float32 HxW or CxHxW (uint16 scaled to 0..1)."""
    raw = np.asarray(raw)
    if raw.dtype not in (np.dtype('float32'), np.dtype('uint16')):
        raise ValueError('Expected Siril float32 or uint16 pixels.')
    if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[0] in (1, 3)):
        data = raw.astype(np.float32)
    else:
        raise ValueError('Only monochrome or channel-first RGB images are supported.')
    if raw.dtype == np.uint16:
        data /= 65535.0
    return np.ascontiguousarray(data), dict(dtype=raw.dtype, shape=raw.shape)


def encode_pixels(data, meta):
    if meta['dtype'] == np.uint16:
        out = np.rint(np.clip(data, 0, 1) * 65535).astype(np.uint16)
    else:
        out = np.asarray(data, np.float32)
    if out.shape != meta['shape']:
        raise ValueError('Output shape does not match the captured image.')
    return np.ascontiguousarray(out)


def fingerprint(raw):
    return raw.shape, raw.dtype.str, hashlib.sha256(np.ascontiguousarray(raw).tobytes()).digest()


def write_result(siril, data, meta, expected, label):
    """Apply to Siril with an Undo state, only if the image is still the one that was processed."""
    final = encode_pixels(data, meta)
    with siril.image_lock():
        current = read_pixels(siril)
        if fingerprint(current) != expected:
            raise RuntimeError('The image in Siril changed since Apply started. Nothing was written.')
        if siril.undo_save_state(label) is False:
            raise RuntimeError('Siril could not save Undo. Pixels were not overwritten.')
        if siril.set_image_pixeldata(final) is False:
            raise RuntimeError('Siril refused the modified pixels.')
        actual = read_pixels(siril)
        if actual.shape != final.shape or not np.allclose(actual, final, rtol=0, atol=0 if final.dtype == np.uint16 else 2e-7, equal_nan=True):
            raise RuntimeError('Siril readback differs from the result. Use Siril Undo if needed.')
    return fingerprint(actual)


def siril_header_text(siril):
    parts = []
    for fn in (lambda: siril.get_image_fits_header(return_as='str'), lambda: ' '.join(str(h) for h in (siril.get_image_history() or [])),
               lambda: siril.get_image_filename()):
        try:
            parts.append(str(fn()))
        except Exception:
            pass
    return ' '.join(parts)


def unique_path(p):
    p = Path(p)
    if not p.exists():
        return p
    for i in range(1, 1000):
        q = p.with_name(f'{p.stem}_{i}{p.suffix}')
        if not q.exists():
            return q
    raise FileExistsError(p)


def prefixed_path(siril, prefix):
    """<prefix><image file name> next to the image Siril has loaded, e.g. starmask_M42.fits (FITS extension kept, others -> .fit;
    never overwrites: _1, _2 ... when it exists).  Falls back to Siril's working folder."""
    name = None
    try:
        name = siril.get_image_filename()
    except Exception:
        pass
    if name and Path(name).parent.is_dir() and str(Path(name).parent) != '.':
        n = Path(name)
        stem = n.name[:-3] if n.name.lower().endswith('.gz') else n.name
        ext = Path(stem).suffix if Path(stem).suffix.lower() in ('.fit', '.fits', '.fts') else '.fit'
        return unique_path(n.with_name(prefix + Path(stem).stem + ext))
    wd = None
    try:
        wd = siril.get_siril_wd()
    except Exception:
        pass
    return unique_path(Path(wd or Path.home()) / f'{prefix}image.fit')


def side_path(siril, suffix):
    """<image>_<suffix>.fit next to the image Siril has loaded (falls back to Siril's working folder)."""
    name = None
    try:
        name = siril.get_image_filename()
    except Exception:
        pass
    if name and Path(name).parent.is_dir() and str(Path(name).parent) != '.':
        return unique_path(Path(name).with_name(Path(name).stem.removesuffix('.fit') + f'_{suffix}.fit'))
    wd = None
    try:
        wd = siril.get_siril_wd()
    except Exception:
        pass
    return unique_path(Path(wd or Path.home()) / f'umbra_{suffix}.fit')


def write_fits(path, data, header=None, history=''):
    from astropy.io import fits
    hdr = header.copy() if header is not None else fits.Header()
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        hdr.pop(k, None)
    if history:
        hdr['HISTORY'] = history
    fits.PrimaryHDU(np.asarray(data, np.float32), header=hdr).writeto(path)


def siril_fits_header(siril):
    try:
        from astropy.io import fits
        return fits.Header.fromstring(siril.get_image_fits_header(return_as='str'), sep='\n')
    except Exception:
        return None


def region_from_selection(siril, minimum=64):
    """Siril's current selection (x, y, w, h) when at least `minimum` px each way, else None."""
    try:
        sel = siril.get_siril_selection()
    except Exception:
        return None
    if sel and sel[2] >= minimum and sel[3] >= minimum:
        return tuple(int(v) for v in sel)
    return None


def crop_region(data, rect=None, size=768):
    """data (H,W) or (C,H,W) -> (cropped view copy, (x, y, w, h)).  rect given: that region; else a centred size x size crop (whole image if smaller)."""
    H, W = data.shape[-2:]
    if rect is None:
        w, h = min(size, W), min(size, H)
        rect = ((W - w) // 2, (H - h) // 2, w, h)
    x, y, w, h = rect
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = min(w, W - x), min(h, H - y)
    return np.ascontiguousarray(data[..., y:y + h, x:x + w]), (x, y, w, h)


# ----------------------------------------------------------------------------------------------- display (auto-stretch)
def _mtf(m, x):
    return (m - 1.0) * x / ((2.0 * m - 1.0) * x - m)


def stretch_params(img, target=0.25, shadow_k=-2.8):
    """Linked auto-stretch (one set of parameters for all channels), like an STF: (c0, midtone)."""
    v = np.asarray(img, np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.5, 1.0
    if v.size > 2_000_000:
        v = v[::max(1, v.size // 2_000_000)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    c0 = min(max(0.0, med + shadow_k * mad), max(med - 1e-6, 0.0)) if mad > 0 else 0.0
    hi = float(np.percentile(v, 99.9)) if v.size else 1.0
    span = max(hi - c0, 1e-6) if hi > c0 else 1.0
    medn = min(max((med - c0) / span, 1e-6), 0.999)
    m = float(_mtf(target, medn))
    return c0, min(max(m, 1e-4), 0.9999), span


def apply_stretch(img, params):
    c0, m, span = params
    x = np.clip((np.nan_to_num(np.asarray(img, np.float32)) - c0) / span, 0, 1)
    y = _mtf(m, x)
    return np.clip(y, 0, 1)


def pool_for_display(img, max_side=1600):
    """Block-mean downsample (C,H,W) so the longer side is <= max_side."""
    H, W = img.shape[-2:]
    f = int(math.ceil(max(H, W) / max_side))
    if f <= 1:
        return img
    h, w = H // f * f, W // f * f
    a = np.nan_to_num(img[..., :h, :w]).reshape(*img.shape[:-2], h // f, f, w // f, f)
    return a.mean((-3, -1)).astype(np.float32)


def to_rgb8(img, params, flip=False):
    """(H,W) or (C,H,W) linear -> (H,W,3) uint8 with the given stretch."""
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = a[None]
    a = apply_stretch(a, params)
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    rgb = np.ascontiguousarray((a.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8))
    return rgb[::-1].copy() if flip else rgb


SOFTNESS = 0.35
REMOVE_RGB, KEEP_RGB = (255, 60, 60), (60, 220, 90)


# ----------------------------------------------------------------------------------------------- brush masks (shared: Starless marks, Restore brush in the other tools)
def smoothstep(a, b, x):
    if b <= a + 1e-8:
        return (x >= b).astype(np.float32)
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def brush_mask(shape, strokes, window=None):
    """Replay portable normalized strokes (points in 0..1 of the image, row-down; size = brush diameter / min(h, w)).
    One stroke has uniform maximum opacity: distance to line segments avoids holes between sparse pointer events and the
    maximum dab coverage within a stroke avoids speed-dependent buildup. Separate strokes alpha-composite, including erasure.
    window=(x0,y0,x1,y1) renders only that part of the full-size mask."""
    h, w = shape
    X0, Y0, X1, Y1 = window if window is not None else (0, 0, w, h)
    canvas = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    sx = max(w - 1, 1)
    sy = max(h - 1, 1)
    for stroke in strokes:
        points = np.asarray(stroke['points'], np.float32) * np.array([sx, sy], np.float32)
        if not len(points):
            continue
        radius = max(.5 * stroke['size'] * min(h, w), .5)
        x0 = max(X0, int(np.floor(points[:, 0].min() - radius)))
        x1 = min(X1, int(np.ceil(points[:, 0].max() + radius)) + 1)
        y0 = max(Y0, int(np.floor(points[:, 1].min() - radius)))
        y1 = min(Y1, int(np.ceil(points[:, 1].max() + radius)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        coverage = np.zeros((y1 - y0, x1 - x0), np.float32)
        segments = zip(points[:-1], points[1:]) if len(points) > 1 else [(points[0], points[0])]
        for a, b in segments:
            xa = max(x0, int(np.floor(min(a[0], b[0]) - radius)))
            xb = min(x1, int(np.ceil(max(a[0], b[0]) + radius)) + 1)
            ya = max(y0, int(np.floor(min(a[1], b[1]) - radius)))
            yb = min(y1, int(np.ceil(max(a[1], b[1]) + radius)) + 1)
            if xb <= xa or yb <= ya:
                continue
            yy, xx = np.ogrid[ya:yb, xa:xb]
            dx, dy = b - a
            den = float(dx * dx + dy * dy)
            t = np.clip(((xx - a[0]) * dx + (yy - a[1]) * dy) / max(den, 1e-12), 0, 1)
            distance = np.sqrt((xx - a[0] - t * dx) ** 2 + (yy - a[1] - t * dy) ** 2)
            if stroke['softness'] <= 1e-6:
                weight = (distance <= radius).astype(np.float32)
            else:
                weight = 1 - smoothstep(radius * (1 - stroke['softness']), radius, distance)
            patch = coverage[ya - y0:yb - y0, xa - x0:xb - x0]
            np.maximum(patch, weight, out=patch)
        alpha = coverage * stroke['opacity']
        patch = canvas[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        if stroke['erase']:
            patch *= 1 - alpha
        else:
            patch += alpha * (1 - patch)
    return np.clip(canvas, 0, 1)


def stroke_layers(strokes, layer):
    """Strokes that make up one mask: its own strokes plus every eraser stroke (the eraser works on both layers, in order)."""
    return [s for s in strokes if s['layer'] == layer or s['erase']]


def has_paint(strokes, layer):
    return any(not s['erase'] for s in stroke_layers(strokes, layer))


def strokes_signature(strokes):
    return hashlib.sha1(json.dumps(strokes, sort_keys=True).encode()).hexdigest()


def brush_fraction(v):
    """Brush slider 1..100 -> stroke 'size' (diameter as a fraction of min(h, w))."""
    return 0.0008 + (v / 100.0) ** 2 * 0.12


def masks_from_strokes(strokes, shape, flip=False):
    """(mask_remove, mask_keep) float32 (H,W) in ARRAY row order from strokes drawn in display coordinates (row 0 = top of the picture).
    flip=True: the picture is the array flipped vertically (Siril shows its bottom-up rows flipped), so the masks are flipped back."""
    out = []
    for layer in ('remove', 'keep'):
        if has_paint(strokes, layer):
            m = brush_mask(shape, stroke_layers(strokes, layer))
            out.append(np.ascontiguousarray(m[::-1] if flip else m, np.float32))
        else:
            out.append(None)
    return out[0], out[1]


def _bilinear(img, mx, my):
    """img (C,H,W); mx, my float pixel coordinates (same shape) -> (C, *mx.shape) samples (edge-clamped)."""
    H, W = img.shape[-2:]
    x0 = np.clip(np.floor(mx).astype(np.int64), 0, W - 1); y0 = np.clip(np.floor(my).astype(np.int64), 0, H - 1)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    fx = np.clip(mx - x0, 0, 1).astype(np.float32); fy = np.clip(my - y0, 0, 1).astype(np.float32)
    a, b, c, d = img[:, y0, x0], img[:, y0, x1], img[:, y1, x0], img[:, y1, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _stroke_window(shape, stroke):
    h, w = shape
    pts = np.asarray(stroke['points'], np.float32) * np.array([max(w - 1, 1), max(h - 1, 1)], np.float32)
    r = max(.5 * stroke['size'] * min(h, w), .5)
    return (max(0, int(np.floor(pts[:, 0].min() - r)) - 1), max(0, int(np.floor(pts[:, 1].min() - r)) - 1),
            min(w, int(np.ceil(pts[:, 0].max() + r)) + 2), min(h, int(np.ceil(pts[:, 1].max() + r)) + 2))


def apply_retouch(result, base, strokes, flip=False):
    """Restore / Clone / Eraser strokes replayed in order on a tool result (C,H,W or H,W, ARRAY row order).  Strokes are in display
    coordinates (row 0 = top of the picture; flip=True: the picture is the array flipped vertically, as Siril shows it).
    Restore blends toward base (the image before the tool); Clone copies from the stroke's source offset (fixed from its first point,
    sampled from the image as it was before that stroke, like Umbra Astro Studio); Eraser blends back to the plain result."""
    if not strokes:
        return result
    r = np.asarray(result, np.float32)
    mono = r.ndim == 2
    R = r[None] if mono else r
    B = np.asarray(base, np.float32)
    B = B[None] if B.ndim == 2 else B
    if B.shape != R.shape:
        B = R
    if flip:
        R, B = R[:, ::-1], B[:, ::-1]
    O = np.array(R, np.float32, copy=True)
    H, W = O.shape[-2:]
    for st in strokes:
        X0, Y0, X1, Y1 = _stroke_window((H, W), st)
        if X1 <= X0 or Y1 <= Y0:
            continue
        one = dict(points=st['points'], size=st['size'], softness=st['softness'], opacity=st['opacity'], erase=False)
        cov = brush_mask((H, W), [one], (X0, Y0, X1, Y1))
        if not cov.any():
            continue
        patch = O[:, Y0:Y1, X0:X1]
        if st.get('erase'):
            target = R[:, Y0:Y1, X0:X1]
        elif st.get('layer') == 'clone' and st.get('source') is not None:
            d = (np.array(st['source'], np.float64) - np.array(st['points'][0], np.float64)) * np.array([max(W - 1, 1), max(H - 1, 1)])
            yy, xx = np.mgrid[Y0:Y1, X0:X1]
            mx, my = xx + d[0], yy + d[1]
            cov = cov * ((mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1))
            target = _bilinear(O, mx, my)
        elif st.get('layer') == 'keep':
            target = B[:, Y0:Y1, X0:X1]
        else:
            continue
        patch += cov[None] * (target - patch)
    if flip:
        O = O[:, ::-1]
    O = np.ascontiguousarray(O)
    return O[0] if mono else O


# ----------------------------------------------------------------------------------------------- UI
STYLE = """
QWidget { background: #242424; color: #dddddd; font-size: 10pt; }
QWidget#Card { background: #303030; border-radius: 10px; }
QLabel, QCheckBox, QRadioButton, QSlider { background: transparent; }
QLabel#Title { font-size: 15pt; font-weight: bold; color: #dddddd; }
QLabel#Badge { color: #888888; font-size: 8pt; }
QLabel#Note { color: #999999; font-size: 9pt; }
QLabel#Value { color: #dddddd; font-weight: bold; min-width: 40px; }
QLabel#Caption { color: #aaaaaa; font-size: 9pt; }
QToolTip { color: #dddddd; background: #404040; border: 1px solid #666666; padding: 6px; }
QCheckBox { spacing: 4px; background: transparent; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #777777; border-radius: 3px; background: #3c3c3c; }
QCheckBox::indicator:checked { background: #1679db; border-color: #6eb5ff; }
QCheckBox::indicator:disabled { border-color: #555555; background: #2e2e2e; }
QComboBox { background: transparent; border: none; padding: 5px 25px 5px 9px; min-height: 18px; combobox-popup: 0; }
QComboBox QAbstractItemView { background: #3d3d3d; color: #dddddd; border: 1px solid #666666; border-radius: 9px; padding: 5px; selection-background-color: #1679db; outline: 0; }
QComboBox QAbstractItemView::item { padding: 5px 8px; border-radius: 5px; }
QPushButton { background: #505050; border: 1px solid #666666; border-radius: 5px; padding: 4px 7px; }
QPushButton:hover { background: #616161; }
QPushButton:checked { background: #1474d5; border-color: #358eef; }
QPushButton:disabled { color: #777777; border-color: #444444; }
QPushButton#Apply { background: #1274d5; color: #ffffff; font-weight: bold; border-color: #338be7; }
QPushButton#Apply:disabled { background: #354656; color: #808890; border-color: #43505c; }
QPushButton#ZoomButton { padding: 3px 10px; }
QScrollArea { border: none; background: #242424; }
QScrollBar:vertical { background: #242424; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #555555; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #242424; height: 8px; margin: 0; }
QScrollBar::handle:horizontal { background: #555555; border-radius: 4px; min-width: 24px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: #505050; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #1474d5; border-radius: 2px; }
QSlider::handle:horizontal { background: #dddddd; width: 14px; margin: -6px 0; border-radius: 7px; }
QSlider::sub-page:horizontal:disabled { background: #555555; }
QProgressBar { border: 1px solid #444444; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: #1474d5; }
"""

MAX_STEPS = 10           # Undo history: the original + this many Apply steps (each a full-size float32 copy)
PANE_SIDE = 2400          # longest side of the images shown in the zoomable pane
IDLE_SIRIL = 'Set the controls, then press Preview or Apply. Apply replaces the image in Siril (Undo available).'
IDLE_TEST = 'Test mode (no Siril).'
IDLE_INLINE = 'Press Apply to see the result on the right, then Apply to Siril to write it (Undo available).'
_QT = {}


def _qt():
    """Build the Qt classes once (PyQt6 is imported only when a window is opened)."""
    if _QT:
        return _QT['ns']
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
    from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QProgressBar,
                                 QComboBox, QCheckBox, QSizePolicy, QStackedWidget, QMenu, QLayout)

    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtGui import QPainter, QPen, QColor
    from PyQt6.QtWidgets import QListView, QFrame, QGraphicsView, QGraphicsScene, QScrollArea
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QCursor, QPainterPath

    class ModernComboBox(QComboBox):
        """Rounded drop-down, drawn the same on every platform (same look as Umbra Astro Studio)."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumHeight(28)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            view = QListView(self); view.setFrameShape(QFrame.Shape.NoFrame); view.setSpacing(1); self.setView(view)
            self.setMaxVisibleItems(100)                 # whole list at once, drops below the box, no scroll arrows
            self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)   # long names are elided, not widening the column
            self.setMinimumContentsLength(14)
            self.setStyleSheet('QComboBox { combobox-popup: 0; }')

        def paintEvent(self, event):
            p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor('#4398ed') if self.hasFocus() else QColor('#666666'), 1))
            p.setBrush(QColor('#606060') if (self.underMouse() and self.isEnabled()) else QColor('#505050'))
            p.drawRoundedRect(QRectF(.5, .5, self.width() - 1, self.height() - 1), 8, 8)
            p.setPen(QColor('#dddddd') if self.isEnabled() else QColor('#888888')); p.setFont(self.font())
            text = self.fontMetrics().elidedText(self.currentText(), Qt.TextElideMode.ElideRight, max(1, self.width() - 35))
            p.drawText(self.rect().adjusted(9, 0, -27, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            cx, cy = self.width() - 15, self.height() / 2
            p.setPen(QPen(QColor('#dddddd') if self.isEnabled() else QColor('#888888'), 1.7))
            p.drawLine(QPointF(cx - 4, cy - 2), QPointF(cx, cy + 2)); p.drawLine(QPointF(cx, cy + 2), QPointF(cx + 4, cy - 2)); p.end()

        def enterEvent(self, event):
            super().enterEvent(event); self.update()

        def leaveEvent(self, event):
            super().leaveEvent(event); self.update()

        def showPopup(self):
            super().showPopup()
            container = self.view().window()
            container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            container.setObjectName('RoundedComboPopup')
            container.setStyleSheet('QFrame#RoundedComboPopup {background:#3d3d3d;border:1px solid #666666;border-radius:10px;padding:4px;} '
                                    'QListView {background:#3d3d3d;color:#dddddd;border:none;border-radius:7px;padding:3px;outline:0;} '
                                    'QListView::item {padding:5px 8px;border-radius:5px;} QListView::item:selected {background:#1679db;}')
            v = self.view()                           # grow the popup until the whole list fits (never a scroll bar)
            want = v.sizeHintForColumn(0) + 40          # wide enough for the longest name
            if want > container.width():
                scr = self.screen().availableGeometry() if self.screen() else None
                w = min(want, scr.right() - container.x() - 4) if scr else want
                container.resize(max(container.width(), w), container.height())
            v.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            for _ in range(4):
                extra = v.verticalScrollBar().maximum()
                if extra <= 0:
                    break
                container.resize(container.width(), container.height() + extra + 2)
                QApplication.processEvents()

    class Worker(QThread):
        done = pyqtSignal(object, object, float, str, object)
        failed = pyqtSignal(str, bool)       # message, during model load
        cancelled = pyqtSignal()
        prog = pyqtSignal(int, int)
        stage = pyqtSignal(str)

        def __init__(self, spec, model_path, data, params, extra=None):
            super().__init__()
            self.spec, self.model_path, self.data, self.params, self.extra = spec, model_path, data, params, extra
            self.cancel_flag = False

        def run(self):
            loading = True
            try:
                t0 = time.time()
                if str(self.model_path) not in [c[0] for c in _ENGINE_CACHE.values()]:
                    self.stage.emit('Loading model…')
                eng = get_engine(self.spec, self.model_path)
                loading = False
                self.stage.emit('Running…')
                out, info = self.spec.engine_run(eng, self.data, self.params, lambda n, t: self.prog.emit(int(n), int(t)), lambda: self.cancel_flag, self.extra)
                self.done.emit(out, info, time.time() - t0, eng.provider, eng.note)
            except Exception as e:
                if not loading and str(e) == 'Cancelled':
                    self.cancelled.emit()
                    return
                import traceback
                tb = traceback.format_exc()
                write_error_log(self.spec, tb)
                self.failed.emit(f'{type(e).__name__}: {e}' if not loading else str(e), loading)

    class Task(QThread):
        done = pyqtSignal(object)
        failed = pyqtSignal(str)
        prog = pyqtSignal(int, int)

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                self.done.emit(self.fn(lambda a, b: self.prog.emit(int(a), int(b))))
            except Exception as e:
                self.failed.emit(str(e) if isinstance(e, DownloadError) else f'{type(e).__name__}: {e}')

    def qimage(rgb):
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    class ZoomView(QGraphicsView):
        """The image area (like Umbra Guided GHS): fits the window until you zoom; wheel / pinch zooms around the mouse, drag pans,
        double-click fits again.  Swapping before / after keeps the zoom and position."""
        MAX_ZOOM = 8.0

        def __init__(self, pane):
            super().__init__()
            self.pane = pane
            self.setScene(QGraphicsScene(self))
            self.item = self.scene().addPixmap(QPixmap())
            self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setFrameShape(QFrame.Shape.NoFrame)
            self.setBackgroundBrush(QColor('#191919'))
            self.setStyleSheet('QGraphicsView { background: #191919; border: 1px solid #484848; border-radius: 7px; }')
            self.setMinimumSize(320, 240)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setToolTip('Scroll or pinch to zoom, drag to pan, double-click to fit. Hold Space (or Show original) for the original.')
            self.fitted = True
            self.placeholder = ''
            # brushes (Starless correction marks): the page that owns the strokes sets them up with set_brush()
            self.overlay = self.scene().addPixmap(QPixmap())
            self.overlay.setZValue(1)
            self.tool = 'navigate'                  # navigate | remove | keep | erase
            self.strokes = None                     # list owned by the tool page, or None = no brushes
            self.brush = dict(size=0.03, opacity=1.0, softness=0.35)
            self.colors = {'remove': (255, 60, 60), 'keep': (60, 220, 90), 'erase': (235, 235, 235), 'clone': (114, 186, 255)}
            self.clone_source = [0.5, 0.5]           # Clone: where the blue ring is (normalized display coordinates); owned by the page
            self.drag_source = False
            self.on_stroke = None
            self.draft = None
            self.cursor_pos = None
            self.mpan = None
            self.max_strokes, self.max_points = 600, 4096
            self.setMouseTracking(True); self.viewport().setMouseTracking(True)

        # ---- brushes
        def set_brush(self, strokes=None, on_stroke=None):
            """strokes: the page's stroke list (None switches the brushes off and hides the marks)."""
            self.strokes, self.on_stroke = strokes, on_stroke
            if strokes is None:
                self.set_tool('navigate')
                self.overlay.setPixmap(QPixmap())

        def set_tool(self, tool):
            self.tool = tool if self.strokes is not None else 'navigate'
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if self.tool == 'navigate' else QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor if self.tool == 'navigate' else Qt.CursorShape.CrossCursor))
            self.viewport().update()

        def set_overlay(self, pm):
            self.overlay.setPixmap(pm if pm is not None else QPixmap())

        def image_shape(self):
            pm = self.item.pixmap()
            return (pm.height(), pm.width())

        def _norm(self, pos):
            h, w = self.image_shape()
            return [float(min(max((pos.x() - 0.5) / max(w - 1, 1), 0), 1)), float(min(max((pos.y() - 0.5) / max(h - 1, 1), 0), 1))]

        def _scene(self, p):
            h, w = self.image_shape()
            return QPointF(p[0] * max(w - 1, 1) + 0.5, p[1] * max(h - 1, 1) + 0.5)

        def _inside(self, pos):
            return not self.item.pixmap().isNull() and QRectF(self.item.pixmap().rect()).contains(pos)

        def brush_radius(self):
            h, w = self.image_shape()
            return max(self.brush['size'] * min(h, w) / 2, 0.5)

        def near_source(self, pos):
            if self.tool != 'clone':
                return False
            c = self._scene(self.clone_source)
            return math.hypot(pos.x() - c.x(), pos.y() - c.y()) <= max(self.brush_radius(), 9 / max(abs(self.transform().m11()), 1e-5))

        def mousePressEvent(self, e):
            if e.button() == Qt.MouseButton.MiddleButton:
                self.mpan = e.position(); e.accept(); return
            if self.tool != 'navigate' and self.strokes is not None and e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                pos = self.mapToScene(e.position().toPoint())
                if e.button() == Qt.MouseButton.LeftButton and self._inside(pos) and self.near_source(pos):   # move the Clone source ring
                    self.drag_source = True
                    self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                    e.accept(); return
                if self._inside(pos) and len(self.strokes) < self.max_strokes:
                    erase = self.tool == 'erase' or e.button() == Qt.MouseButton.RightButton
                    self.draft = dict(layer='both' if erase else self.tool, erase=bool(erase), size=float(self.brush['size']),
                                      softness=float(self.brush['softness']), opacity=float(self.brush['opacity']), points=[self._norm(pos)])
                    if self.tool == 'clone' and not erase:
                        self.draft['source'] = [round(v, 7) for v in self.clone_source]
                    self.strokes.append(self.draft)
                    self.cursor_pos = pos
                    self.viewport().update()
                e.accept(); return
            super().mousePressEvent(e)

        def mouseMoveEvent(self, e):
            if self.mpan is not None:
                d = e.position() - self.mpan; self.mpan = e.position()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
                e.accept(); return
            pos = self.mapToScene(e.position().toPoint())
            self.cursor_pos = pos if self._inside(pos) else None
            if self.drag_source:
                self.clone_source[:] = self._norm(pos)
                self.viewport().update(); e.accept(); return
            if self.tool == 'clone' and self.draft is None:
                self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor if (self.cursor_pos is not None and self.near_source(pos))
                                                  else Qt.CursorShape.CrossCursor))
            if self.draft is not None:
                if self._inside(pos) and len(self.draft['points']) < self.max_points:
                    last = self._scene(self.draft['points'][-1])
                    if abs(pos.x() - last.x()) + abs(pos.y() - last.y()) >= 1.0:
                        self.draft['points'].append([round(v, 7) for v in self._norm(pos)])
                self.viewport().update(); e.accept(); return
            if self.tool != 'navigate':
                self.viewport().update(); e.accept(); return
            super().mouseMoveEvent(e)

        def mouseReleaseEvent(self, e):
            if e.button() == Qt.MouseButton.MiddleButton:
                self.mpan = None; e.accept(); return
            if self.drag_source:
                self.drag_source = False
                self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                self.viewport().update(); e.accept(); return
            if self.draft is not None:
                self.draft = None
                self.viewport().update()
                if self.on_stroke is not None:
                    self.on_stroke()
                e.accept(); return
            super().mouseReleaseEvent(e)

        def leaveEvent(self, e):
            self.cursor_pos = None
            self.viewport().update()
            super().leaveEvent(e)

        def set_pixmap(self, pm, keep_view=True):
            same = (not self.item.pixmap().isNull()) and self.item.pixmap().size() == pm.size()
            self.item.setPixmap(pm)
            self.scene().setSceneRect(QRectF(pm.rect()))
            if not (keep_view and same) or self.fitted:
                self.fit()
            self.viewport().update()

        def clear(self, text=''):
            self.item.setPixmap(QPixmap())
            self.placeholder = text
            self.viewport().update()

        def fit(self):
            pm = self.item.pixmap()
            self.resetTransform()
            if not pm.isNull():
                self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
            self.fitted = True

        def zoom_by(self, factor):
            pm = self.item.pixmap()
            if pm.isNull():
                return
            vp = self.viewport()
            current = max(abs(self.transform().m11()), 1e-9)
            fitted = min(max(1, vp.width() - 4) / max(pm.width(), 1), max(1, vp.height() - 4) / max(pm.height(), 1))
            floor, ceiling = min(fitted, 1.0), max(self.MAX_ZOOM / max(self.devicePixelRatioF(), 1e-6), fitted)
            factor = min(max(factor, floor / current), ceiling / current)
            self.scale(factor, factor)
            self.fitted = abs(self.transform().m11() - fitted) < 1e-3 * fitted

        def wheelEvent(self, e):
            if self.draft is not None:
                e.accept(); return
            delta = e.angleDelta().y() or e.pixelDelta().y()
            if delta:
                self.zoom_by(1.2 if delta > 0 else 1 / 1.2)
            e.accept()

        def viewportEvent(self, e):
            if e.type() == QEvent.Type.NativeGesture and e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self.zoom_by(math.exp(float(e.value())))
                return True
            return super().viewportEvent(e)

        def mouseDoubleClickEvent(self, e):
            if self.tool == 'navigate':
                self.fit()
            e.accept()

        def resizeEvent(self, e):
            super().resizeEvent(e)
            if self.fitted:
                self.fit()

        def keyPressEvent(self, e):
            self.pane.keyPressEvent(e)

        def keyReleaseEvent(self, e):
            self.pane.keyReleaseEvent(e)

        def drawForeground(self, p, rect):
            if self.item.pixmap().isNull() and self.placeholder:
                p.save(); p.resetTransform()
                p.setPen(QColor('#9a9a9a'))
                p.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.placeholder)
                p.restore()
            h, w = self.image_shape()
            if self.draft is not None:          # live trail while the mouse is down (the marks image is rebuilt on release)
                pts = [self._scene(q) for q in self.draft['points']]
                pen = QPen(QColor(*self.colors['erase' if self.draft['erase'] else self.draft['layer']], int(60 + 80 * self.draft['opacity'])
                                  if self.draft['layer'] != 'clone' else 40))
                pen.setWidthF(max(self.draft['size'] * min(h, w), 1.0))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap); pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.save(); p.setPen(pen)
                if len(pts) == 1:
                    p.drawPoint(pts[0])
                else:
                    path = QPainterPath(pts[0])
                    for q in pts[1:]:
                        path.lineTo(q)
                    p.drawPath(path)
                p.restore()
            if self.tool == 'clone' and not self.item.pixmap().isNull():
                self.draw_clone(p)
                return
            if self.tool != 'navigate' and self.cursor_pos is not None:
                rad = max(self.brush['size'] * min(h, w) / 2, 0.5)
                p.save(); p.setRenderHint(QPainter.RenderHint.Antialiasing)
                pen = QPen(QColor('#111111'), 3); pen.setCosmetic(True); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(self.cursor_pos, rad, rad)
                pen = QPen(QColor(*self.colors[self.tool]), 1); pen.setCosmetic(True); p.setPen(pen)
                p.drawEllipse(self.cursor_pos, rad, rad)
                p.restore()

        def draw_clone(self, p):
            """Clone (like Umbra Astro Studio): the blue source ring (drag it), and under the brush a ghost of the patch it will copy."""
            rad = self.brush_radius()
            scale = max(abs(self.transform().m11()), 1e-5)
            src = self._scene(self.clone_source)
            anchor = src
            if self.draft is not None and self.draft.get('source') is not None and self.cursor_pos is not None:
                first = self._scene(self.draft['points'][0])
                anchor = QPointF(src.x() + self.cursor_pos.x() - first.x(), src.y() + self.cursor_pos.y() - first.y())
            p.save(); p.setRenderHint(QPainter.RenderHint.Antialiasing)
            for centre in ([src] if anchor is src else [src, anchor]):
                pen = QPen(QColor('#141414'), 3); pen.setCosmetic(True); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(centre, rad, rad)
                pen = QPen(QColor('#72baff'), 1); pen.setCosmetic(True); pen.setStyle(Qt.PenStyle.DashLine); p.setPen(pen)
                p.drawEllipse(centre, rad, rad)
                p.drawLine(centre + QPointF(-5 / scale, 0), centre + QPointF(5 / scale, 0))
                p.drawLine(centre + QPointF(0, -5 / scale), centre + QPointF(0, 5 / scale))
            c = self.cursor_pos
            if c is not None and not self.drag_source and (self.draft is not None or not self.near_source(c)):
                clip = QPainterPath(); clip.addEllipse(c, rad, rad)
                p.save(); p.setClipPath(clip); p.setOpacity(0.85 * float(self.brush['opacity']))
                pm = self.item.pixmap()
                p.drawPixmap(QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad), pm, QRectF(anchor.x() - rad, anchor.y() - rad, 2 * rad, 2 * rad))
                p.restore()
                pen = QPen(QColor('#111111'), 3); pen.setCosmetic(True); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(c, rad, rad)
                pen = QPen(QColor('#eeeeee'), 1); pen.setCosmetic(True); p.setPen(pen)
                p.drawEllipse(c, rad, rad)
                pen.setStyle(Qt.PenStyle.DotLine); p.setPen(pen)
                inner = rad * (1 - float(self.brush['softness']))
                p.drawEllipse(c, inner, inner)
            p.restore()

    class PreviewWindow(QWidget):
        """The image pane on the right of the window (shared by all tools): the original before a run, then before / after with the
        same linked auto-stretch; hold Space (or Show original) for the original; zoom / pan like Umbra Guided GHS."""
        def __init__(self, spec=None, embedded=True):
            super().__init__()
            self.spec, self.embedded = spec, True
            self.pix = {}
            self.size_note = ''
            self.titles = {'before': 'Original', 'after': 'Result', 'alt': 'Gradient model'}
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 12, 14, 12)
            lay.setSpacing(6)
            row = QHBoxLayout()
            row.setSpacing(6)
            self.undo_btn = QPushButton('Undo')      # like Umbra Astro Studio: step back / forward through Apply results (all tools)
            self.undo_btn.setToolTip('Undo the last Apply (any tool). Cmd/Ctrl+Z. The image in Siril changes only with Apply to Siril.')
            self.redo_btn = QPushButton('Redo')
            self.redo_btn.setToolTip('Redo the step that was undone. Cmd/Ctrl+Shift+Z.')
            self.undo_btn.setEnabled(False); self.redo_btn.setEnabled(False)
            row.addWidget(self.undo_btn); row.addWidget(self.redo_btn)
            row.addSpacing(6)
            self.caption = QLabel('', objectName='Badge')
            row.addWidget(self.caption)
            row.addStretch(1)
            self.alt_btn = QPushButton('Show gradient model')
            self.alt_btn.setCheckable(True)
            self.alt_btn.setToolTip('Shows the gradient model: the background that is subtracted. Press again to go back to the result.')
            self.alt_btn.toggled.connect(lambda c: (self.hold(False) if c else None, self.refresh()))
            self.alt_btn.hide()
            row.addWidget(self.alt_btn)
            self.fit_btn = QPushButton('Fit', objectName='ZoomButton')
            self.fit_btn.setToolTip('Fit the whole image in the window (or double-click the image).')
            self.fit_btn.clicked.connect(lambda: self.view.fit())
            row.addWidget(self.fit_btn)
            self.orig_btn = QPushButton('Show original')
            self.show_orig = False
            self.orig_btn.setToolTip('Hold the button (or the Space key) to see the original; release to go back to the result.')
            self.orig_btn.pressed.connect(lambda: self.hold(True))
            self.orig_btn.released.connect(lambda: self.hold(False))
            self.orig_btn.setEnabled(False)
            row.addWidget(self.orig_btn)
            lay.addLayout(row)
            self.view = ZoomView(self)
            lay.addWidget(self.view, 1)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def set_alt(self, label):
            """label: text of the alternative-view button (e.g. 'Show gradient model'), or None for tools without one."""
            self.alt_label = label
            if label:
                self.alt_btn.setText(label)
                self.titles['alt'] = label[5:].capitalize() if label.startswith('Show ') else label
            if not label:
                self.alt_btn.setChecked(False)
                self.alt_btn.hide()

        def hold(self, flag):
            self.show_orig = bool(flag) and self.orig_btn.isEnabled()
            self.refresh()

        def show_state(self, state):
            """state: dict saved by a tool page (see snapshot), or None for an empty pane."""
            if not state:
                self.pix = {}
                self.orig_btn.setEnabled(False)
                self.alt_btn.setChecked(False); self.alt_btn.hide()
                self.caption.setText('')
                self.view.clear('No image loaded in Siril.')
                return
            self.pix = dict(state['pix'])
            self.size_note = state.get('note', '')
            self.orig_btn.setEnabled(state['result'])
            has_alt = 'alt' in self.pix and bool(getattr(self, 'alt_label', None))
            self.alt_btn.setVisible(has_alt)
            if not has_alt:
                self.alt_btn.setChecked(False)
            self.show_orig = False
            self.caption.setText(state['caption'])
            self.refresh(keep_view=state.get('keep_view', True))

        def show_result(self, before, after, alt, caption):
            """before / after / alt: (H,W,3) uint8 (alt may be None)."""
            st = dict(pix={'before': QPixmap.fromImage(qimage(before)), 'after': QPixmap.fromImage(qimage(after))}, result=True, caption=caption,
                      note=self.size_note, keep_view=False)
            if alt is not None:
                st['pix']['alt'] = QPixmap.fromImage(qimage(alt))
            self.show_state(st)
            return st

        def show_original(self, before, caption):
            """Before any run: the original only."""
            p = QPixmap.fromImage(qimage(before))
            st = dict(pix={'before': p, 'after': p}, result=False, caption=caption, note='', keep_view=False)
            self.show_state(st)
            return st

        def current_key(self):
            if self.alt_btn.isVisible() and self.alt_btn.isChecked() and 'alt' in self.pix:
                return 'alt'
            return 'before' if self.show_orig else 'after'

        def refresh(self, keep_view=True):
            if len(self.pix) > 1 and self.orig_btn.isEnabled():
                self.caption.setText(self.titles[self.current_key()] + self.size_note)
            p = self.pix.get(self.current_key())
            if p is None or p.isNull():
                return
            self.view.set_pixmap(p, keep_view=keep_view)

        def keyPressEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.hold(True)
            else:
                super().keyPressEvent(e)

        def keyReleaseEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.hold(False)
            else:
                super().keyReleaseEvent(e)

    class Ui:
        """Control builders handed to the tool panels so every tool looks the same."""
        def __init__(self, view):
            self.view = view

        def slider(self, layout, text, lo, hi, default, fmt, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            val = QLabel(fmt(default), objectName='Value')
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val)
            layout.addLayout(row)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi); s.setValue(default)
            s.valueChanged.connect(lambda v, l=val: l.setText(fmt(v)))
            if tip:
                label.setToolTip(tip); s.setToolTip(tip); val.setToolTip(tip)
            layout.addWidget(s)
            s.label_widget, s.value_widget = label, val
            return s

        def check(self, layout, text, tip=None, checked=False):
            c = QCheckBox(text)
            c.setChecked(checked)
            if tip:
                c.setToolTip(tip)
            layout.addWidget(c)
            return c

        def combo(self, layout, text, items, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            cb = ModernComboBox()
            for t, v in items:
                cb.addItem(t, v)
            if tip:
                label.setToolTip(tip); cb.setToolTip(tip)
            row.addWidget(cb, 1)
            layout.addLayout(row)
            cb.label_widget = label
            return cb

        def note(self, layout, text):
            l = QLabel(text, objectName='Note', wordWrap=True)
            layout.addWidget(l)
            return l

        @staticmethod
        def show(widget, flag):
            for w in (widget, getattr(widget, 'label_widget', None), getattr(widget, 'value_widget', None)):
                if w is not None:
                    w.setVisible(bool(flag))
            QTimer.singleShot(0, lambda: widget.window().adjustSize())

        @staticmethod
        def enable(widget, flag):
            for w in (widget, getattr(widget, 'label_widget', None), getattr(widget, 'value_widget', None)):
                if w is not None:
                    w.setEnabled(bool(flag))

        def model_dir(self):
            return model_dir(self.view.spec)[0]

    class BrushCard:
        """Second card for Denoise / Deblur / Gradient: a Restore brush paints back the image the tool started from (the result is
        blended toward it under the marks, no new AI run).  Greyed out until Apply has made a result."""
        RESTORE_TIP = 'Paint where the result should go back to the image before this tool. Click again to go back to panning.'

        def __init__(self, ui, host):
            self.host = host
            self.card2 = QWidget(objectName='Card')
            self.card2.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            layout = QVBoxLayout(self.card2)
            layout.setContentsMargins(12, 10, 12, 12)
            self.ready_note = ui.note(layout, 'Press Apply above first. Then paint with Restore to bring back the image before this tool.')
            ui.note(layout, 'Restore paints back the image from before this tool. Clone copies from the blue ring: drag the ring '
                            'onto a clean area, then paint over what you want to cover. Eraser (or right-click) takes marks away. '
                            'Click a pressed brush again to pan. [ and ] change the brush size; hold Space to see the original.')
            row = QHBoxLayout(); row.setSpacing(4)
            self.tool_btns = {}
            for key, text, tip in (('keep', 'Restore', self.RESTORE_TIP),
                                   ('clone', 'Clone', 'Copy from the blue source ring (drag it to move it) onto where you paint, like Umbra Astro '
                                                      'Studio. Click again to go back to panning.'),
                                   ('erase', 'Eraser', 'Takes Restore and Clone marks away (or right-click with any brush). Click again to go back to panning.')):
                b = QPushButton(text); b.setCheckable(True); b.setToolTip(tip); b.setMinimumHeight(26)
                b.clicked.connect(lambda checked, k=key: self.set_tool(k if checked else 'navigate'))
                row.addWidget(b); self.tool_btns[key] = b
            layout.addLayout(row)
            self.brush = ui.slider(layout, 'Brush size', 1, 100, 30, lambda v: f'{v}', 'Brush diameter. Keys [ and ] make it smaller / bigger.')
            self.opacity = ui.slider(layout, 'Opacity', 5, 100, 100, lambda v: f'{v}%',
                                     'Strength of each new stroke: 100% brings the original back fully, lower values partly.')
            row2 = QHBoxLayout()
            self.show_marks = ui.check(row2, 'Show marks', 'Show the painted marks over the image.', True)
            self.clear_btn = QPushButton('Clear marks'); self.clear_btn.setToolTip('Removes all marks.')
            self.clear_btn.clicked.connect(self.clear_marks)
            row2.addStretch(1); row2.addWidget(self.clear_btn)
            layout.addLayout(row2)
            self.strokes, self.pane, self.pending, self.ready = [], None, False, False
            self.clone_source = [0.5, 0.5]
            self.timer = QTimer(); self.timer.setSingleShot(True); self.timer.timeout.connect(self.rerun)
            self.brush.valueChanged.connect(lambda v: self.sync_brush())
            self.opacity.valueChanged.connect(lambda v: self.sync_brush())
            self.show_marks.toggled.connect(lambda _=False: self.update_overlay())
            self.set_ready(False)

        def set_ready(self, flag):
            self.ready = bool(flag)
            self.card2.setEnabled(self.ready)
            self.ready_note.setVisible(not self.ready)
            if not self.ready:
                self.set_tool('navigate')

        def reset(self):
            """The image this tool starts from changed (another step, Undo...): marks belong to the old one."""
            self.strokes.clear()
            self.set_ready(False)
            self.update_overlay()

        def pane_activated(self, pane):
            self.pane = pane
            pane.view.set_brush(self.strokes, self.stroke_done)
            pane.view.clone_source = self.clone_source
            pane.view.set_tool(self.current_tool())
            self.sync_brush()
            self.update_overlay()

        def current_tool(self):
            return next((k for k, b in self.tool_btns.items() if b.isChecked()), 'navigate')

        def set_tool(self, key):
            for k, b in self.tool_btns.items():
                b.setChecked(k == key)
            if self.pane is not None and self.pane.view.strokes is self.strokes:
                self.pane.view.set_tool(key)

        def sync_brush(self):
            if self.pane is not None and self.pane.view.strokes is self.strokes:
                self.pane.view.brush.update(size=brush_fraction(self.brush.value()), opacity=self.opacity.value() / 100.0, softness=SOFTNESS)
                self.pane.view.viewport().update()

        def update_overlay(self):
            if self.pane is None or self.pane.view.strokes is not self.strokes:
                return
            v = self.pane.view
            h, w = v.image_shape()
            if not self.strokes or not self.show_marks.isChecked() or h < 2 or w < 2 or not has_paint(self.strokes, 'keep'):
                v.set_overlay(None)
                return
            m = brush_mask((h, w), stroke_layers(self.strokes, 'keep')) * 0.45
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = KEEP_RGB
            rgba[..., 3] = np.clip(m * 255, 0, 255).astype(np.uint8)
            q = QImage(np.ascontiguousarray(rgba).data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
            v.set_overlay(QPixmap.fromImage(q))

        def stroke_done(self):
            self.update_overlay()
            if self.ready:
                self.timer.start(150)

        def rerun(self):
            if self.host.busy:
                self.pending = True
                return
            if self.host.model_path():
                self.host.start('preview')

        def after_run(self):
            self.set_ready(True)
            self.update_overlay()
            if self.pending:
                self.pending = False
                self.timer.start(50)

        def clear_marks(self):
            had = bool(self.strokes)
            self.strokes.clear()
            self.update_overlay()
            if had and self.ready:
                self.timer.start(50)

        def apply(self, out, base, flip):
            """Restore / Clone / Eraser strokes replayed on the result."""
            if not self.strokes:
                return out
            return apply_retouch(out, base, self.strokes, flip)

        def label(self):
            parts = [n for k, n in (('keep', 'restore'), ('clone', 'clone')) if any(s['layer'] == k and not s['erase'] for s in self.strokes)]
            return f" ({' + '.join(parts)} brush)" if parts else ''

        def signature(self):
            return strokes_signature(self.strokes) if self.strokes else ''

        def settings(self):
            return dict(brush=self.brush.value(), opacity=self.opacity.value())

        def set_settings(self, d):
            for k, w in (('brush', self.brush), ('opacity', self.opacity)):
                if k in (d or {}):
                    try:
                        w.setValue(int(d[k]))
                    except Exception:
                        pass

    class ToolView(QWidget):
        """One tool: title, model card + tool controls, Preview / Apply, progress, status, Close. Used alone or as a page of the combined window."""
        def __init__(self, spec, siril=None, fake_image=None, prefs=None, upd=None, show_title=True, smoke=False, pane=None):
            super().__init__()
            self.spec, self.siril, self.fake, self.prefs, self.smoke = spec, siril, fake_image, prefs, smoke
            self.upd = upd or updspec_for(spec)
            self.worker = self.net = self.preview = None
            self.info = None
            self.job = None
            self.cache = None            # (identity, params key, out, info) of the last whole-image preview (Gradient)
            self.last_preview = None     # (before, after, alt) uint8 of the last preview
            self.result = None
            self.last_error = None
            self.busy = False
            if siril is not None and callable(getattr(spec, 'on_siril', None)):
                try:
                    spec.on_siril(siril)
                except Exception:
                    pass
            saved_model, saved_settings = (prefs.load(spec.key) if prefs else (None, {}))
            self.saved_model = saved_model
            self.left = self                          # the controls column; the image pane on the right belongs to the window
            lay = QVBoxLayout(self)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(8)
            self.lay = lay
            if show_title:
                lay.addWidget(QLabel(spec.title, objectName='Title'))
                lay.addWidget(QLabel(f'v{VERSION}', objectName='Badge'))
            for n in spec.notes:
                lay.addWidget(QLabel(n, objectName='Note', wordWrap=True))
            card = QWidget(objectName='Card')
            card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 10, 12, 12)
            mrow = QHBoxLayout()
            mrow.addWidget(QLabel('Model'))
            self.model_box = ModernComboBox()
            mrow.addWidget(self.model_box, 1)
            self.dl_btn = QPushButton('Download')        # downloads (or updates) the model selected on the left
            self.dl_btn.setMinimumHeight(28)
            self.dl_btn.setToolTip("Download the selected model from this project's GitHub releases (checked against a SHA-256 before it is installed).")
            self.dl_btn.clicked.connect(self.download_clicked)
            mrow.addWidget(self.dl_btn)
            cl.addLayout(mrow)
            self.upd_note = QLabel('', objectName='Note', wordWrap=True)   # result of the update check made at every start
            self.upd_note.setOpenExternalLinks(True)
            self.upd_note.setTextFormat(Qt.TextFormat.RichText)
            self.upd_note.hide()
            cl.addWidget(self.upd_note)
            self.top_apply_row = None
            if spec.apply_top and spec.inline_preview:   # Apply under the model row (left); the tool controls go in a second card
                self.top_apply_row = QHBoxLayout()
                cl.addSpacing(4); cl.addLayout(self.top_apply_row)
            self.ui = Ui(self)
            self.panel = spec.build_panel(self.ui, cl, self)
            self.panel.set_settings(saved_settings)
            lay.addWidget(card)
            self.brushes = None
            card2 = getattr(self.panel, 'card2', None)
            if card2 is None and spec.restore_brush:
                self.brushes = BrushCard(self.ui, self)
                self.brushes.set_settings((saved_settings or {}).get('_brush', {}))
                card2 = self.brushes.card2
            if card2 is not None:
                lay.addWidget(card2)
            brow2 = QHBoxLayout()
            self.preview_btn = QPushButton('Preview')
            self.preview_btn.setToolTip('Runs on the selection in Siril (or a centred crop) and shows the result here. The image is not changed.' if not spec.preview_whole
                                        else 'Runs on the whole image and shows the result here, downsampled. The image is not changed.')
            self.preview_btn.clicked.connect(lambda: self.start('preview'))
            self.apply_btn = QPushButton('Apply', objectName='Apply')
            self.apply_btn.clicked.connect(lambda: self.start('apply'))
            if spec.inline_preview:                   # Apply (small blue, inside the card) shows the result; Apply to Siril (beside Close) writes it
                self.preview_btn.setText('Apply')
                self.preview_btn.setObjectName('Apply')
                self.preview_btn.setMinimumWidth(90)
                self.preview_btn.setToolTip('Runs on the whole image and shows the result on the right. The image in Siril is not changed.')
                self.apply_btn.setText('Apply to Siril')
                self.apply_btn.setToolTip('Writes the result to the image in Siril (Undo available). Reuses the result shown on the right when nothing has changed.')
                if self.top_apply_row is not None:
                    self.top_apply_row.addStretch(1); self.top_apply_row.addWidget(self.preview_btn)
                else:
                    arow = QHBoxLayout()
                    arow.addStretch(1); arow.addWidget(self.preview_btn)
                    cl.addSpacing(4); cl.addLayout(arow)
            else:
                brow2.addWidget(self.preview_btn); brow2.addWidget(self.apply_btn)
            self.edit_btn = None
            if spec.has_editor:                       # Starless: Keep / Remove brushes in their own window
                self.edit_btn = QPushButton('Edit marks')
                self.edit_btn.setToolTip('Opens the editor for correction marks (Remove / Keep brushes) on the whole image.')
                self.edit_btn.clicked.connect(lambda: self.start('edit'))
                brow2.addWidget(self.edit_btn)
            if brow2.count():
                lay.addLayout(brow2)
            prow = QHBoxLayout()
            self.progress = QProgressBar()
            self.progress.setRange(0, 0); self.progress.setTextVisible(False); self.progress.hide()
            self.cancel_btn = QPushButton('Cancel')
            self.cancel_btn.clicked.connect(self.cancel)
            self.cancel_btn.hide()
            prow.addWidget(self.progress, 1); prow.addWidget(self.cancel_btn)
            lay.addLayout(prow)
            self.idle_msg = IDLE_SIRIL if siril is not None else IDLE_TEST
            self.status = QLabel(self.idle_msg, objectName='Note', wordWrap=True)
            lay.addWidget(self.status)
            close = QPushButton('Close')
            close.clicked.connect(lambda: self.window().close())
            self.undo_btn = QPushButton('Undo all')
            self.undo_btn.setToolTip('Puts back the image as it was when this window opened (Siril Undo still works afterwards).')
            self.undo_btn.clicked.connect(self.undo_all)
            self.undo_btn.setEnabled(False)
            if spec.inline_preview:
                crow = QHBoxLayout()
                crow.addWidget(close); crow.addWidget(self.undo_btn); crow.addWidget(self.apply_btn)
                lay.addLayout(crow)
            else:
                lay.addWidget(close)
            self.idle_msg = (IDLE_INLINE if siril is not None else IDLE_TEST) if spec.inline_preview else self.idle_msg
            self.status.setText(self.idle_msg)
            self.model_box.currentIndexChanged.connect(lambda *_: self.model_changed())
            self.folder_note = None
            lay.addStretch(1)
            self.preview = pane                       # shared image pane (MainWindow)
            self.pane_state = None                    # what this page last showed in the pane: (image identity, state)
            self.refresh_models()

        def activate(self):
            """This page became the current one: show its last result in the pane if it belongs to the current image, else the original."""
            w = self.window()
            if self.preview is None:
                return
            self.preview.set_alt(self.spec.alt[0] if self.spec.alt else None)
            hook = getattr(self.panel, 'pane_activated', None) if self.brushes is None else self.brushes.pane_activated
            if hook is not None:
                hook(self.preview)
            else:
                self.preview.view.set_brush(None)
            if self.pane_state is not None and self.pane_state[0] == getattr(w, 'head_id', lambda: None)():
                self.preview.show_state(dict(self.pane_state[1], keep_view=True))
            else:
                getattr(w, 'show_current', lambda: None)()
            getattr(w, 'check_view', lambda v: None)(self)

        # ---- status
        def set_status(self, msg, error=False):
            self.status.setText(msg)
            self.status.setStyleSheet('color: #e05050;' if error else '')
            if self.isVisible():                      # a wrapped status line may need more (or less) height
                QTimer.singleShot(0, self.window().adjustSize)

        def model_path(self):
            d = self.model_box.currentData()
            return d if isinstance(d, str) else None      # a '(Not Downloaded)' entry holds its manifest entry instead of a path

        def selected_entry(self):
            """(manifest entry, state) for the selected dropdown item, state 'missing' / 'update' / 'current', or (None, None)."""
            d = self.model_box.currentData()
            if isinstance(d, dict):
                return d, 'missing'
            if isinstance(d, str) and self.info:
                for m, st in self.info['models']:
                    if not m.get('kind') and Path(d).name == m['file'] and Path(d).parent == Path(download_dir(self.spec)):
                        return m, st
            return None, None

        def other_downloads(self):
            """Manifest items other than the selected model that can be installed or updated (e.g. the Auto-keep model)."""
            if not self.info:
                return []
            sel = self.selected_entry()[0]
            offer = getattr(self.spec, 'offer_extras', True)
            return [(m, st) for m, st in self.info['models'] if st != 'current' and m is not sel and ((m.get('kind') and offer) or (st == 'update' and not m.get('kind')))]

        def update_dl_button(self, busy=None):
            busy = self.busy if busy is None else busy
            e, st = self.selected_entry()
            if st == 'missing':
                self.dl_btn.setText('Download'); self.dl_btn.setEnabled(not busy)
                self.dl_btn.setToolTip(f"Download {e['name']} ({e['size_bytes'] / 1e6:.0f} MB). Checked against a SHA-256 before it is installed.")
            elif st == 'update':
                self.dl_btn.setText('Update'); self.dl_btn.setEnabled(not busy)
                self.dl_btn.setToolTip(f"Download the published version of {e['name']} ({e['size_bytes'] / 1e6:.0f} MB).")
            else:
                more = self.other_downloads()
                self.dl_btn.setText('Download'); self.dl_btn.setEnabled(bool(more) and not busy)
                self.dl_btn.setToolTip('Other files to download: ' + ', '.join(m['name'] for m, _ in more) if more
                                       else 'The selected model is installed.')
            hot = st in ('missing', 'update') or not self.model_box.count()
            self.dl_btn.setStyleSheet('background: #1274d5; color: #ffffff; font-weight: bold;' if hot and not busy else '')

        def save_prefs(self):
            if self.prefs:
                st = dict(self.panel.settings())
                if getattr(self, 'brushes', None) is not None:
                    st['_brush'] = self.brushes.settings()
                self.prefs.save(self.spec.key, self.model_path(), st, getattr(self.window(), 'current_key', None))

        def set_busy(self, flag, net=False):
            self.busy = flag
            has = bool(self.model_path())
            self.preview_btn.setEnabled(not flag and has); self.apply_btn.setEnabled(not flag and has)
            self.update_dl_button(busy=flag)
            if getattr(self, 'edit_btn', None) is not None:
                self.edit_btn.setEnabled(not flag and has)
            self.model_box.setEnabled(not flag)
            getattr(self.window(), 'update_history_buttons', lambda: None)()
            self.progress.setVisible(flag); self.cancel_btn.setVisible(flag and not net)
            if flag:
                self.progress.setRange(0, 0)

        # ---- models
        def refresh_models(self, select=None):
            sp = self.spec
            cur = select or self.model_path() or default_model_path(sp, self.saved_model)
            self.model_box.blockSignals(True)
            self.model_box.clear()
            items = list_models(sp)
            for label, path in items:
                self.model_box.addItem(label, path)
            i = self.model_box.findData(cur) if cur else -1
            if i < 0 and cur and Path(cur).is_file():
                self.model_box.addItem(model_label(sp, Path(cur).stem), cur)
                i = self.model_box.count() - 1
            n_installed = self.model_box.count()
            for m, st in (self.info or {}).get('models', []):    # published models not installed yet (the label loses the tag once installed)
                if st == 'missing' and not m.get('kind'):
                    self.model_box.addItem(f"{model_label(sp, Path(m['file']).stem)} (Not Downloaded)", m)
                    if i < 0 and not n_installed and m.get('default'):
                        i = self.model_box.count() - 1
            if self.model_box.count():
                self.model_box.setCurrentIndex(max(i, 0))
            self.model_box.blockSignals(False)
            has = bool(self.model_path())
            self.preview_btn.setEnabled(has and not self.busy); self.apply_btn.setEnabled(has and not self.busy)
            self.update_dl_button()
            d, fb = model_dir(sp)
            if fb and self.folder_note is None:
                self.folder_note = f"Using models from the old '{sp.fallback_folder}' folder."
            self.model_changed(remember=bool(select))

        def model_changed(self, *_, remember=False):
            path = self.model_path()
            self.update_dl_button()
            has = bool(path)
            self.preview_btn.setEnabled(has and not self.busy); self.apply_btn.setEnabled(has and not self.busy)
            if not path:
                self.panel.model_changed(None)
                e, st = self.selected_entry()
                if st == 'missing':
                    self.set_status(f"{e['name']} is not downloaded yet. Click Download to install it ({e['size_bytes'] / 1e6:.0f} MB).")
                else:
                    self.set_status(missing_model_message(self.spec, self.info))
                return
            self.panel.model_changed(path)
            if remember:
                self.save_prefs()
            idle = self.idle_msg
            if self.folder_note:
                idle = self.folder_note + ' ' + idle
            self.set_status(idle)

        # ---- updates / downloads (a button press starts every network action, except the manifest lookup at launch)
        def run_net(self, fn, on_done, busy_text):
            if self.net is not None and self.net.isRunning():
                return
            self.net = Task(fn)
            self.net.prog.connect(self.net_progress)
            self.net.done.connect(lambda r: (self.set_busy(False), on_done(r)))
            self.net.failed.connect(lambda m: (self.set_busy(False), self.net_failed(m)))
            self.set_busy(True, net=True)
            self.set_status(busy_text)
            self.net.start()

        def net_progress(self, got, total):
            if total > 0:
                self.progress.setRange(0, 1000); self.progress.setValue(int(1000 * got / total))
                self.set_status(f'Downloading… {got / 1e6:.0f} / {total / 1e6:.0f} MB')

        def net_failed(self, msg):
            self.set_status(msg + ('' if self.model_path() else ' No model is installed yet.'), True)

        def fetch_info(self, manual=False, then=None):
            sp = self.spec
            if offline_mode(sp):
                self.upd_note.setText(f'Offline mode ({sp.prefix}OFFLINE is set); using the installed models.'); self.upd_note.show()
                return
            def work(prog):
                return check_updates(fetch_manifest(self.upd, use_cache=not manual), sp)
            def done(res):
                self.info = res
                self.show_update_note(res)
                keep = self.model_box.currentData()
                self.refresh_models(select=keep if isinstance(keep, str) else None)
                if then:
                    then()
            def failed(msg):
                self.info = None
                self.upd_note.setStyleSheet('')
                self.upd_note.setText(msg); self.upd_note.show()
                self.set_status(msg if not self.model_path() else self.idle_msg, not self.model_path())
            self.net = Task(work)
            self.net.done.connect(lambda r: (self.set_busy(False), done(r)))
            self.net.failed.connect(lambda m: (self.set_busy(False), failed(m)))
            self.set_busy(True, net=True)
            self.set_status('Checking the model list…')
            self.net.start()

        def show_update_note(self, res):
            """Green when the script and this tool's models are up to date, yellow when something new is available."""
            from html import escape
            lines = []
            if res['script_newer']:
                url = res['script_url']
                lines.append(f'There is an update to this script available ({escape(res["script_version"])}, you have {VERSION}).'
                             + (f' <a href="{escape(url)}" style="color: #e8c547;">Release page</a>' if url else ''))
            if any(st == 'missing' and not m.get('kind') for m, st in res['models']):
                lines.append('There is a new model available for this tool, select the model in the dropdown and click Download to install.')
            if any(st == 'update' and not m.get('kind') for m, st in res['models']):
                lines.append('An update is available for a model you have: select it in the dropdown and click Update.')
            if lines:
                self.upd_note.setStyleSheet('color: #e8c547;')
                self.upd_note.setText('<br>'.join(lines))
            else:
                self.upd_note.setStyleSheet('color: #5cc96b;')
                self.upd_note.setText('The models for this tool are up to date.')
            self.upd_note.show()

        def download_clicked(self):
            e, st = self.selected_entry()
            if st in ('missing', 'update'):
                self.start_download(e)
                return
            more = self.other_downloads()
            if not more:
                self.update_dl_button()
                return
            if len(more) == 1:
                self.start_download(more[0][0])
                return
            menu = QMenu(self)
            for m, st2 in more:
                tag = 'Update ' if st2 == 'update' else 'Download '
                menu.addAction(f"{tag}{m['name']} ({m['size_bytes'] / 1e6:.0f} MB)").triggered.connect(lambda _=False, e=m: self.start_download(e))
            menu.exec(self.dl_btn.mapToGlobal(self.dl_btn.rect().bottomLeft()))

        def show_dl_menu(self):
            self.download_clicked()

        def start_download(self, entry):
            def work(prog):
                return download_model(entry, self.upd, download_dir(self.spec), progress=prog)
            def done(path):
                if self.info:                       # the downloaded file now matches the published one
                    self.info = dict(self.info, models=[(m, 'current' if m is entry else st) for m, st in self.info['models']])
                    self.show_update_note(self.info)
                if entry.get('kind'):               # an extra file (e.g. the Auto-keep model): not a model for the picker
                    getattr(self.panel, 'extras_changed', lambda: None)()
                    self.update_dl_button()
                    self.set_status(f"Installed {entry['name']}. " + self.idle_msg)
                    return
                self.refresh_models(select=str(path))
                self.set_status(f"Installed {entry['name']}. " + self.idle_msg)
            self.run_net(work, done, f"Downloading {entry['name']}…")

        # ---- run
        def cancel(self):
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel_flag = True
                self.cancel_btn.setEnabled(False)
            hook = getattr(self.panel, 'cancel', None)
            if hook is not None:
                hook()

        def undo_all(self):
            getattr(self.window(), 'undo_all', lambda v: None)(self)

        def read_image(self, kind):
            """-> (data, meta, identity, header_text): the image this tool starts from (the window's current image, see MainWindow.base_for)."""
            w = self.window()
            b = w.base_for(self.spec.key) if hasattr(w, 'base_for') else None
            if b is None:
                raise RuntimeError('No image loaded in Siril.')
            return b['data'], w.meta, ('h', b['id']), w.htxt

        def job_key(self):
            """Same key start() builds for a run with the current settings."""
            params = dict(self.panel.params())
            params.update(self.spec.image_params(getattr(self.window(), 'htxt', '')))
            params = self.spec.preview_params(params)
            return json.dumps(params, sort_keys=True, default=str) + str(self.model_path()) + self.brush_sig()

        def brush_sig(self):
            return self.brushes.signature() if self.brushes is not None else ''

        def start(self, kind):
            if self.busy:
                return
            if kind == 'apply' and self.spec.inline_preview and hasattr(self.window(), 'apply_to_siril'):
                self.window().apply_to_siril(self)       # writes the window's current image (all tools' Apply steps) to Siril
                return
            if getattr(self.panel, 'editor_busy', lambda: False)():
                self.set_status('The preview window is still working. Wait until it has finished.')
                return
            sp = self.spec
            path = self.model_path()
            if not path:
                self.set_status(missing_model_message(sp, self.info), True)
                return
            try:
                data, meta, identity, htxt = self.read_image(kind)
            except Exception as e:
                self.set_status(f'Could not read the image: {e}', True)
                return
            params = dict(self.panel.params())
            params.update(sp.image_params(htxt))
            if kind == 'edit' and sp.has_editor:
                self.save_prefs()
                self.panel.open_editor(self, data, meta, identity, params, path)
                return
            rect, region, crop = None, 'whole image', None
            full = data
            if kind == 'preview':
                params = sp.preview_params(params)
                if not sp.preview_whole:
                    sel = region_from_selection(self.siril) if self.siril is not None else None
                    H, W = full.shape[-2:]
                    # Siril selections are in display coordinates; pixel rows are bottom-up (rows H-y-h .. H-y)
                    _, arect = crop_region(full, None if sel is None else (sel[0], H - sel[1] - sel[3], sel[2], sel[3]))
                    rect = sel if sel is not None else arect
                    x, y, w, h = arect
                    data = np.ascontiguousarray(full[..., y:y + h, x:x + w])
                    if w < 512 or h < 512:                   # fixed 512 px models: give small regions context, crop back afterwards
                        tw, th = min(max(w, 640), W), min(max(h, 640), H)
                        x0 = min(max(x - (tw - w) // 2, 0), W - tw); y0 = min(max(y - (th - h) // 2, 0), H - th)
                        if (tw, th) != (w, h):
                            data = np.ascontiguousarray(full[..., y0:y0 + th, x0:x0 + tw])
                            crop = (x - x0, y - y0, w, h)
                    region = (f'selection {rect[2]}×{rect[3]} px at {rect[0]},{rect[1]}' if sel is not None
                              else (f'centre crop {rect[2]}×{rect[3]} px' if (rect[2], rect[3]) != (W, H) else f'whole image {W}×{H} px'))
                else:
                    region = f'whole image {data.shape[-1]}×{data.shape[-2]} px'
            key = json.dumps(params, sort_keys=True, default=str) + str(path)
            if (kind == 'apply' or (kind == 'preview' and self.brushes is not None)) and sp.reuse_preview and self.cache and identity is not None and self.cache[0] == identity and self.cache[1] == key:
                self.job = dict(kind=kind, meta=meta, identity=identity, params=params, region=region, data=data, key=key, htxt=htxt)
                self.finish(self.cache[2], self.cache[3], 0.0, self.cache[4], self.cache[5], reused=True)
                return
            if sp.linear_only:
                w = stretched_warning(data)
                if w:
                    msg = f'{sp.name}: Warning: {w}'
                    if self.siril is not None:
                        try:
                            self.siril.log(msg)
                        except Exception:
                            pass
                    else:
                        print(msg, flush=True)
            self.job = dict(kind=kind, meta=meta, identity=identity, params=params, region=region, data=data, key=key, htxt=htxt, rect=rect, crop=crop, full=full)
            try:
                extra = sp.prepare_job(self, self.job)
            except Exception as e:
                self.set_status(str(e), True)
                return
            self.result = None; self.last_error = None
            self.save_prefs()
            self.set_busy(True)
            self.cancel_btn.setEnabled(True)
            self.set_status('Loading model…' if str(path) not in [c[0] for c in _ENGINE_CACHE.values()] else 'Running…')
            self.worker = Worker(sp, path, data, params, extra)
            self.worker.stage.connect(self.set_status)
            self.worker.prog.connect(self.on_progress)
            self.worker.done.connect(self.finish)
            self.worker.failed.connect(self.fail)
            self.worker.cancelled.connect(self.was_cancelled)
            self.worker.start()

        def on_progress(self, n, total):
            if total > 0:
                self.progress.setRange(0, total); self.progress.setValue(n)
                self.set_status(f'Running… {self.spec.progress_unit} {n} / {total}')

        def was_cancelled(self):
            self.set_busy(False)
            self.set_status('Cancelled. The image was not changed.')

        def fail(self, msg, loading=False):
            self.set_busy(False)
            self.last_error = msg
            self.set_status((f'Could not load the model: {msg}' if loading else msg), True)

        def finish(self, out, info, secs, provider, note, reused=False):
            sp, job = self.spec, self.job
            self.set_busy(False)
            extra = ' '.join(dict.fromkeys(info.get('notes') or []))
            if note and note not in extra:
                extra = (extra + ' ' + note).strip()
            try:
                if job['kind'] == 'preview':
                    if job.get('crop'):                       # run on a larger context region; show only the selected part
                        cx, cy, cw, ch = job['crop']
                        sl = (Ellipsis, slice(cy, cy + ch), slice(cx, cx + cw))
                        out = out[sl]
                        job = dict(job, data=job['data'][sl])
                        info = dict(info, **{k: np.asarray(v)[sl] for k, v in info.items() if isinstance(v, np.ndarray) and v.ndim >= 2})
                    raw = out
                    lab = f"{sp.name} {sp.label(job['params'])}"
                    if self.brushes is not None:
                        out = self.brushes.apply(out, job['data'], self.siril is not None)
                        lab += self.brushes.label()
                    st = self.show_preview(out, info, job)
                    w = self.window()
                    if hasattr(w, 'commit'):
                        w.commit(sp.key, sp.output(job['params'], out, info), lab, job['params'], info, job['key'] + self.brush_sig())
                        self.pane_state = (w.head_id(), st)
                    out = raw
                    QTimer.singleShot(0, self.brushes.after_run if self.brushes is not None else getattr(self.panel, 'after_run', lambda: None))
                    if getattr(self, 'pending_write', False):
                        self.pending_write = False
                        QTimer.singleShot(0, lambda: w.write_head(self))
                    if sp.reuse_preview and job['identity'] is not None:
                        self.cache = (job['identity'], job['key'], out, info, provider, note)
                    self.result = out
                    tail = (' Press Apply to Siril to write it.' if sp.inline_preview else ' The image was not changed.')
                    self.set_status(f'{"Done" if sp.inline_preview else "Preview done"} in {secs:.1f} s. Ran on {provider}.' + tail + (' ' + extra if extra else ''))
                    return
                label = f"{sp.name} {VERSION} {sp.label(job['params'])}"
                write_out = sp.output(job['params'], out, info)
                msg = ''
                ctx = types.SimpleNamespace(siril=self.siril, header_text=job['htxt'], history=label, view=self)
                if self.siril is not None:
                    self.cache = None
                    self.job['identity'] = write_result(self.siril, write_out, job['meta'], job['identity'], label)
                    msg = sp.post_apply(ctx, job['params'], info) or ''
                    self.siril.log(f'{label} done in {secs:.1f} s (Undo available).' + (' ' + msg if msg else ''))
                H0, W0 = job['data'].shape[-2:]
                if not sp.preview_whole and min(H0, W0) < 512:
                    extra = (extra + ' Images smaller than 512 px give less accurate results near the edges.').strip()
                self.result = write_out
                self.set_status(f'Done in {secs:.1f} s. Ran on {provider}. ' + ('Applied to Siril; use Siril Undo to revert.' if self.siril is not None else 'Test mode: nothing was written.')
                                + (' ' + msg if msg else '') + (' ' + extra if extra else ''))
            except Exception as e:
                self.fail(str(e))

        def show_preview(self, out, info, job):
            sp = self.spec
            before = pool_for_display(job['data'], PANE_SIDE)
            after = pool_for_display(out, PANE_SIDE)
            flip = self.siril is not None
            sp_before = stretch_params(before)
            b8, a8 = to_rgb8(before, sp_before, flip), to_rgb8(after, sp_before, flip)
            alt8 = None
            if sp.alt and info.get(sp.alt[1]) is not None:
                alt = pool_for_display(np.asarray(info[sp.alt[1]], np.float32), PANE_SIDE)
                alt8 = to_rgb8(alt, sp_before if getattr(sp, 'alt_same_stretch', False) else stretch_params(alt), flip)
            self.last_preview = (b8, a8, alt8)
            w = self.window()
            self.preview.set_alt(sp.alt[0] if sp.alt else None)
            self.preview.size_note = f' · {job["data"].shape[-1]}×{job["data"].shape[-2]} px'
            st = self.preview.show_result(b8, a8, alt8, f'Preview of the {job["region"]}. Same auto-stretch for before and after.')
            self.pane_state = (None, st)
            return st

        def shutdown(self):
            for t in (self.worker, self.net):
                if t is not None and t.isRunning():
                    if t is self.worker:
                        t.cancel_flag = True
                    t.wait(8000)
            getattr(self.panel, 'shutdown', lambda: None)()

    class MainWindow(QWidget):
        """Like Umbra Guided GHS: a fixed-width controls column on the left (scrolls if a tool needs more room) and one zoomable image
        pane on the right that shows the original until a tool has run.  The window keeps its size when you switch tools or models."""
        LEFT_W = 400

        def __init__(self, specs, combined, siril=None, fake_image=None, smoke=False):
            super().__init__()
            self.specs = {s.key: s for s in specs}
            self.combined = combined
            self.siril, self.fake = siril, fake_image
            self.setStyleSheet(STYLE)
            self.views = {}
            # the working image: every Apply adds a step; all tools start from the current step; Apply to Siril writes it
            self.hist, self.hidx, self.next_id = [], 0, 1
            self.meta, self.htxt, self.siril_fp, self.written_id = None, '', None, 0
            self._cur = None
            outer = QHBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            left = QWidget()
            left.setFixedWidth(self.LEFT_W)
            ll = QVBoxLayout(left)
            ll.setContentsMargins(0, 0, 0, 0)
            ll.setSpacing(0)
            content = QWidget()
            lay = QVBoxLayout(content)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            scroll = QScrollArea()
            scroll.setWidget(content)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            ll.addWidget(scroll)
            outer.addWidget(left)
            self.pane = PreviewWindow()
            outer.addWidget(self.pane, 1)
            if combined:
                self.setWindowTitle('Umbra Astro AI (BETA)')
                prefs = Prefs(SCRIPT_DIR / 'Umbra_Astro_AI_prefs.json', combined=True, enabled=not smoke)
                top = QWidget()
                tl = QVBoxLayout(top)
                tl.setContentsMargins(14, 12, 14, 0)
                tl.setSpacing(4)
                tl.addWidget(QLabel('Umbra Astro AI (BETA)', objectName='Title'))
                tl.addWidget(QLabel(f'v{VERSION}', objectName='Badge'))
                row = QHBoxLayout()
                row.addWidget(QLabel('Tool'))
                self.tool_box = ModernComboBox()
                for s in specs:
                    self.tool_box.addItem(s.name, s.key)
                row.addWidget(self.tool_box, 1)
                tl.addLayout(row)
                lay.addWidget(top)
                self.top = top
                self.stack = None
                for s in specs:
                    v = ToolView(s, siril, fake_image, prefs, combined_upd(specs), show_title=False, smoke=smoke, pane=self.pane)
                    self.views[s.key] = v
                    lay.addWidget(v)
                    v.hide()
                self.prefs = prefs
                first = prefs.load_tool()
                self.current_key = first if first in self.views else specs[0].key
                self.tool_box.setCurrentIndex(self.tool_box.findData(self.current_key))
                self.tool_box.currentIndexChanged.connect(lambda *_: self.switch(self.tool_box.currentData()))
            else:
                s = specs[0]
                self.setWindowTitle(s.title)
                self.current_key = s.key
                v = ToolView(s, siril, fake_image, Prefs(SCRIPT_DIR / f'{s.script}_prefs.json', enabled=not smoke), show_title=True, smoke=smoke, pane=self.pane)
                self.views[s.key] = v
                lay.addWidget(v)
            # one width for every tool page (the widest one), so switching tools never moves the image pane
            need = max([v.minimumSizeHint().width() for v in self.views.values()] + [self.LEFT_W]) + 10
            left.setFixedWidth(need)
            self.left_w = need
            avail = QApplication.primaryScreen().availableGeometry()   # fit the screen, like Umbra Guided GHS
            self.resize(min(1400, avail.width() - 40), min(900, avail.height() - 40))
            self.setMinimumSize(self.left_w + 360, 520)
            QApplication.instance().installEventFilter(self)       # Space (hold) and [ ] work wherever the keyboard focus is
            self.pane.undo_btn.clicked.connect(self.undo)
            self.pane.redo_btn.clicked.connect(self.redo)
            QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self, self.undo)
            QShortcut(QKeySequence(QKeySequence.StandardKey.Redo), self, self.redo)
            self.switch(self.current_key, initial=True)
            QTimer.singleShot(0, self.load_original)

        def view(self):
            return self.views[self.current_key]

        def adjustSize(self):
            """Tool pages ask for a refit when rows appear / disappear or a status wraps: the window keeps its size (the left column scrolls)."""
            if self.layout() is not None:
                self.layout().activate()

        def refit(self):
            self.adjustSize()

        # ---- the working image and its history (Undo / Redo / Undo all / Apply to Siril)
        def read_source(self):
            """-> (data, meta, fingerprint, header text) of the image in Siril (or the test image)."""
            if self.siril is not None:
                with self.siril.image_lock():
                    raw = read_pixels(self.siril)
                    data, meta = decode_pixels(raw)
                    return data, meta, fingerprint(raw), siril_header_text(self.siril)
            f = self.fake
            return f, dict(dtype=np.dtype('float32'), shape=f.shape), fingerprint(f), ''

        def load_original(self):
            """Read the image once when the window opens; it is step 0 of the history."""
            try:
                data, meta, fp, htxt = self.read_source()
            except Exception:
                self.hist = []
                self.pane.show_state(None)
                return
            arr = np.array(data, np.float32, copy=True)
            arr.setflags(write=False)
            self.hist = [dict(id=0, data=arr, label='Original', tool=None, params=None, info=None, key=None, open=False, posted=True)]
            self.hidx, self.meta, self.htxt, self.siril_fp, self.written_id = 0, meta, htxt, fp, 0
            self.view().activate()
            self.update_history_buttons()

        def head(self):
            return self.hist[self.hidx] if self.hist else None

        def head_id(self):
            h = self.head()
            return None if h is None else h['id']

        def base_index(self, key):
            """A tool re-applying (new settings, slider, brush) replaces its own latest step; otherwise it starts from the current step."""
            i = self.hidx
            return i - 1 if (i > 0 and self.hist[i]['tool'] == key and self.hist[i]['open']) else i

        def base_for(self, key):
            return self.hist[self.base_index(key)] if self.hist else None

        def open_step(self, key):
            return self.hist[self.hidx] if (self.hist and self.base_index(key) < self.hidx) else None

        def busy_any(self):
            return any(v.busy for v in self.views.values())

        def commit(self, key, data, label, params, info, jkey):
            arr = np.array(data, np.float32, copy=True)
            arr.setflags(write=False)
            e = dict(id=self.next_id, data=arr, label=label, tool=key, params=params, info=info, key=jkey, open=True, posted=False)
            self.next_id += 1
            if self.base_index(key) < self.hidx:
                self.hist[self.hidx] = e
            else:
                for h in self.hist:
                    h['open'] = False
                del self.hist[self.hidx + 1:]
                self.hist.append(e)
                self.hidx += 1
            while len(self.hist) > MAX_STEPS + 1 and self.hidx > 1:     # memory: keep the original + the last MAX_STEPS steps
                del self.hist[1]
                self.hidx -= 1
            self.update_history_buttons()

        def history_changed(self):
            for v in self.views.values():
                v.cache = None; v.result = None; v.last_preview = None; v.pane_state = None
                if getattr(v, 'brushes', None) is not None:
                    v.brushes.set_ready(False)
            for h in self.hist:
                h['open'] = False
            self.view().activate()
            self.update_history_buttons()

        def undo(self):
            if not self.hist or self.hidx == 0 or self.busy_any():
                return
            gone = self.hist[self.hidx]['label']
            self.hidx -= 1
            self.history_changed()
            self.view().set_status(f'Undo: removed {gone}. Redo brings it back. The image in Siril changes only with Apply to Siril.')

        def redo(self):
            if not self.hist or self.hidx >= len(self.hist) - 1 or self.busy_any():
                return
            self.hidx += 1
            self.history_changed()
            self.view().set_status(f"Redo: {self.hist[self.hidx]['label']}. Press Apply to Siril to write it.")

        def can_undo_all(self):
            return bool(self.hist) and (self.hidx > 0 or self.written_id != 0)

        def update_history_buttons(self):
            busy = self.busy_any()
            self.pane.undo_btn.setEnabled(bool(self.hist) and self.hidx > 0 and not busy)
            self.pane.redo_btn.setEnabled(bool(self.hist) and self.hidx < len(self.hist) - 1 and not busy)
            for v in self.views.values():
                v.undo_btn.setEnabled(self.can_undo_all() and not busy)

        def undo_all(self, view):
            """Back to the image as it was when the window opened, in the window and (if something was written) in Siril. Redo still works."""
            if not self.can_undo_all() or self.busy_any():
                return
            wrote = False
            if self.written_id != 0:
                if self.siril is not None:
                    try:
                        self.siril_fp = write_result(self.siril, self.hist[0]['data'], self.meta, self.siril_fp, f'Umbra {VERSION} undo all')
                        self.siril.log('Umbra: Undo all - the image is back to how it was when the window opened.')
                        wrote = True
                    except Exception as e:
                        view.set_status(f'Undo all could not put the original back in Siril: {e}', True)
                        return
                self.written_id = 0
            self.hidx = 0
            self.history_changed()
            self.view().set_status('Undo all: back to the image as it was when this window opened' + (', also in Siril.' if wrote else '.')
                                   + (' Redo brings the steps back.' if len(self.hist) > 1 else ''))

        def apply_to_siril(self, view):
            if not self.hist or self.busy_any():
                return
            op = self.open_step(view.spec.key)
            if op is not None and view.model_path() and op['key'] != view.job_key():   # settings changed since this tool's Apply: run, then write
                view.pending_write = True
                view.start('preview')
                return
            if self.hidx == 0 and self.written_id == 0:
                view.set_status('Nothing to write yet. Press Apply to see the result, then Apply to Siril.')
                return
            self.write_head(view)

        def write_head(self, view):
            h = self.head()
            if h is None:
                return
            if h['id'] == self.written_id:
                view.set_status('The image in Siril already is the current result.')
                return
            chain = self.hist[1:self.hidx + 1]
            label = ('Umbra: ' + ' + '.join(e['label'] for e in chain)) if chain else 'Umbra: back to the original'
            msgs = []
            if self.siril is not None:
                try:
                    self.siril_fp = write_result(self.siril, h['data'], self.meta, self.siril_fp, f'{label} ({VERSION})')
                except Exception as e:
                    view.set_status(f'Apply to Siril failed: {e}', True)
                    return
                for e in chain:                                   # side outputs (Starless: starmask_<file>) once per step
                    if e['posted']:
                        continue
                    sv = self.views.get(e['tool'])
                    if sv is not None:
                        ctx = types.SimpleNamespace(siril=self.siril, header_text=self.htxt, history=e['label'], view=sv)
                        try:
                            m = sv.spec.post_apply(ctx, e['params'], e['info'] or {})
                        except Exception as ex:
                            m = f'{sv.spec.name}: could not save its extra file ({ex}).'
                        if m:
                            msgs.append(m)
                    e['posted'] = True
                try:
                    self.siril.log(f'{label}: applied (Siril Undo available).' + (' ' + ' '.join(msgs) if msgs else ''))
                except Exception:
                    pass
            self.written_id = h['id']
            self.update_history_buttons()
            view.set_status(('Applied to Siril (Siril Undo also works).' if self.siril is not None else 'Test mode: nothing was written.')
                            + (' ' + ' '.join(msgs) if msgs else ''))

        def brush_slider(self):
            v = self.view()
            if getattr(v, 'brushes', None) is not None:
                return v.brushes.brush if v.brushes.ready else None
            b = getattr(v.panel, 'brush', None)
            return b if (b is not None and b.isEnabled()) else None

        def eventFilter(self, obj, e):
            t = e.type()
            if t in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and self.isActiveWindow():
                from PyQt6.QtWidgets import QLineEdit, QAbstractSpinBox
                if isinstance(QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox)):
                    return False
                k = e.key()
                if k == Qt.Key.Key_Space:
                    if not e.isAutoRepeat():
                        self.pane.hold(t == QEvent.Type.KeyPress)
                    return True
                if k in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight) and t == QEvent.Type.KeyPress:
                    s = self.brush_slider()
                    if s is not None:
                        step = max(1, round(s.value() * 0.15))
                        s.setValue(s.value() + (step if k == Qt.Key.Key_BracketRight else -step))
                    return True
            return False

        def show_current(self):
            """The pane shows the current image (the original, or the result of the steps so far)."""
            h = self.head()
            if h is None:
                self.pane.show_state(None)
                return
            if self._cur is None or self._cur[0] != h['id']:
                full = h['data']
                b = pool_for_display(full, PANE_SIDE)
                b8 = to_rgb8(b, stretch_params(b), self.siril is not None)
                p = QPixmap.fromImage(qimage(b8))
                cap = (f'Original · {full.shape[-1]}×{full.shape[-2]} px' if self.hidx == 0
                       else f"Current image · {self.hidx} step{'s' if self.hidx > 1 else ''} · {full.shape[-1]}×{full.shape[-2]} px")
                self._cur = (h['id'], dict(pix={'before': p, 'after': p}, result=False, caption=cap, note=''))
            self.pane.set_alt(None)
            self.pane.show_state(dict(self._cur[1], keep_view=True))

        def show_original(self):
            self.show_current()

        def check_view(self, v):
            """Tool pages that adapt to the image they start from (Deblur: no stars -> Automatic PSF greyed out) check it in the background."""
            b = self.base_for(v.spec.key)
            fn, cb = getattr(v.panel, 'image_check', None), getattr(v.panel, 'image_checked', None)
            if b is not None and getattr(v, 'brushes', None) is not None and getattr(v, 'brush_base', None) != b['id']:
                v.brush_base = b['id']
                v.brushes.reset()
            if b is None or fn is None or cb is None or getattr(v, 'checked_id', None) == b['id']:
                return
            v.checked_id = b['id']
            self._checks = [t for t in getattr(self, '_checks', []) if t.isRunning()]
            t = Task(lambda prog, fn=fn, d=b['data']: fn(d))
            t.done.connect(lambda r, cb=cb: cb(r))
            t.failed.connect(lambda m: None)
            self._checks.append(t)
            t.start()

        def switch(self, key, initial=False):
            if not initial:
                self.view().save_prefs()
                unload_engines()
            self.current_key = key
            for k, v in self.views.items():
                v.setVisible(k == key)
            self.adjustSize()
            if not initial:
                self.views[key].activate()
                self.view().save_prefs()

        def closeEvent(self, e):
            for t in getattr(self, '_checks', []):
                t.wait(10000)
            for v in self.views.values():
                v.shutdown()
            try:
                self.view().save_prefs()
            except Exception:
                pass
            super().closeEvent(e)

    _QT['ns'] = types.SimpleNamespace(**{k: v for k, v in locals().items() if k in (
        'Worker', 'Task', 'PreviewWindow', 'ZoomView', 'Ui', 'ToolView', 'MainWindow', 'QApplication', 'QTimer', 'ModernComboBox', 'qimage')})
    return _QT['ns']


def fake_image(seed=0, size=192):
    """Synthetic linear RGB frame (noise + a few stars + a gentle gradient) for --ui-test."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    img = (0.02 + 0.003 * rng.standard_normal((3, size, size)) + (0.01 * xx / size)[None]).astype(np.float32)
    for cy, cx in ((40, 50), (110, 120), (150, 40), (70, 150)):
        img += (0.3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.6 ** 2)))[None].astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def run_app(specs, combined, siril=None, fake=None, smoke=False, shots=None):
    """Open the window. smoke: run Preview once for every tool on the fake image, report, exit (return code 0 = all fine)."""
    q = _qt()
    app = q.QApplication.instance() or q.QApplication(sys.argv)
    win = q.MainWindow(specs, combined, siril, fake, smoke)
    ts = os.environ.get('UMBRA_TEST_WINDOW', '')          # tests only: window size 'WxH'
    if smoke and re.fullmatch(r'\d+x\d+', ts):
        win.resize(*map(int, ts.split('x')))
    win.show()
    if not smoke:
        for v in win.views.values():
            if v is win.view():
                q.QTimer.singleShot(200, lambda v=v: (v.fetch_info(manual=False) if not offline_mode(v.spec) else None))
        if combined:
            win.tool_box.currentIndexChanged.connect(lambda *_: (win.view().info is None and not offline_mode(win.view().spec)) and win.view().fetch_info(manual=False))
        return app.exec()
    state = dict(i=0, ok=True, t0=time.time())
    keys = [s.key for s in specs]
    outdir = Path(shots) if shots else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    def snap(v, key):
        if not outdir:
            return
        pre = f'Umbra_Astro_AI_{key}' if combined else win.view().spec.script
        win.grab().save(str(outdir / f'{pre}_panel.png'))
        ed = getattr(v.panel, 'ed', None)
        if ed is not None and ed.isVisible():
            ed.grab().save(str(outdir / f'{pre}_editor.png'))
        if v.preview is not None and v.preview.isVisible():
            v.preview.grab().save(str(outdir / f'{pre}_preview.png'))
            if v.preview.alt_btn is not None and v.preview.alt_btn.isVisible():
                v.preview.alt_btn.setChecked(True)
                v.preview.grab().save(str(outdir / f'{pre}_preview_{v.spec.alt[1]}.png'))
                v.preview.alt_btn.setChecked(False)

    def next_tool():
        if state['i'] >= len(keys):
            print('UI TEST', 'OK' if state['ok'] else 'FAILED', flush=True)
            win.close(); app.quit()
            return
        key = keys[state['i']]
        if combined:
            win.tool_box.setCurrentIndex(win.tool_box.findData(key))
        v = win.view()
        if not v.model_path():
            # no model: Preview/Apply must be disabled and the status must say so
            good = (not v.preview_btn.isEnabled() and not v.apply_btn.isEnabled() and 'No model installed' in v.status.text())
            print(f'UI TEST {key}: no model installed; controls {"OK" if good else "BAD"}: {v.status.text()}', flush=True)
            state['ok'] &= good
            app.processEvents(); snap(v, key)
            state['i'] += 1
            q.QTimer.singleShot(50, next_tool)
            return
        if outdir:                                   # the pane before any run: the original
            app.processEvents()
            win.grab().save(str(outdir / f"{f'Umbra_Astro_AI_{key}' if combined else v.spec.script}_before.png"))
        v.start('preview')
        state['t0'] = time.time()
        q.QTimer.singleShot(250, lambda: poll(v, key))

    def poll(v, key):
        if v.busy and time.time() - state['t0'] < 900:
            q.QTimer.singleShot(250, lambda: poll(v, key))
            return
        ed = getattr(v.panel, 'ed', None)
        ok = v.last_error is None and v.result is not None and bool(np.isfinite(v.result).all()) and v.last_preview is not None
        if ok:
            v.preview.orig_btn.pressed.emit(); ok &= v.preview.current_key() == 'before'; v.preview.orig_btn.released.emit(); ok &= v.preview.current_key() == 'after'
        print(f'UI TEST {key}: {"OK" if ok else "FAILED"} {v.status.text()}', flush=True)
        state['ok'] &= ok
        app.processEvents()
        snap(v, key)
        if ed is not None:
            ed.shutdown()
        state['i'] += 1
        q.QTimer.singleShot(50, next_tool)
    q.QTimer.singleShot(100, next_tool)
    app.exec()
    return 0 if state['ok'] else 1


# ----------------------------------------------------------------------------------------------- CLI
def read_fits(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        hdu = next(x for x in h if x.data is not None)
        data = np.asarray(hdu.data, np.float32)
        header = hdu.header.copy()
        text = ' '.join(repr(x.header) for x in h) + ' ' + ' '.join(str(x.name) for x in h) + ' ' + Path(path).name
    return data, header, text


def run_file(spec, a):
    src, dst = Path(a.image).resolve(), Path(a.out).resolve()
    if dst == src:
        raise SystemExit('Refusing to overwrite the input file (--out equals --image).')
    if dst.exists():
        raise SystemExit(f'Output exists, refusing to overwrite: {dst}')
    data, header, text = read_fits(src)
    params = spec.params_from_args(a)
    params.update(spec.image_params(text))
    if spec.linear_only:
        w = stretched_warning(data)
        if w:
            print(f'{spec.name}: Warning: {w}', flush=True)
    path = a.model or default_model_path(spec)
    if not path:
        raise SystemExit(f'No model installed. Run with --download, or put a .onnx file in {model_dir(spec)[0]}.')
    t0 = time.time()
    eng = get_engine(spec, path)
    out, info = spec.engine_run(eng, data, params, None, None, None)
    label = f"{spec.name} {VERSION} {spec.label(params)}"
    write_out = spec.output(params, out, info)
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        header.pop(k, None)
    header['HISTORY'] = label
    from astropy.io import fits
    fits.PrimaryHDU(np.asarray(write_out, np.float32), header=header).writeto(dst)
    msg = spec.post_cli(a, dst, info, header, label)
    print(f'wrote {dst} shape={write_out.shape} finite={bool(np.isfinite(write_out).all())} device={eng.provider} model={Path(path).name} '
          f'{time.time() - t0:.1f} s' + (f' {msg}' if msg else '') + (f' NOTE: {" ".join(info.get("notes") or [])}' if info.get('notes') else '')
          + (f' NOTE: {eng.note}' if eng.note else ''))


def cli_updates(specs, a, upd_for=None):
    """--check-updates / --download for one or several tools. Returns an exit code."""
    try:
        for spec in specs:
            upd = upd_for(spec) if upd_for else updspec_for(spec)
            if offline_mode(spec):
                print(f'Offline mode ({spec.prefix}OFFLINE is set); nothing to check.')
                continue
            res = check_updates(fetch_manifest(upd), spec)
            if len(specs) > 1:
                print(f'== {spec.name}')
            if a.check_updates:
                print('\n'.join(describe_updates(res, spec)))
            if a.download:
                want = [e for e, st in res['models'] if st != 'current' and (a.download == 'all' or (a.download == 'default' and e.get('default')) or a.download == e['id'])]
                if not want and len(specs) == 1:
                    print('Nothing to download (installed and up to date, or unknown model id).')
                for e in want:
                    last = [0]
                    def pr(g, t, e=e, last=last):
                        if g - last[0] >= 20_000_000 or g == t:
                            last[0] = g; print(f"  {e['name']}: {g / 1e6:.0f} / {t / 1e6:.0f} MB", flush=True)
                    print('installed', download_model(e, upd, download_dir(spec), pr))
        return 0
    except DownloadError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1


def main(specs, combined=False, doc=''):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', help='standalone: FITS file to process (never modified)')
    ap.add_argument('--out', help='standalone: output FITS (must not exist)')
    ap.add_argument('--model', help='model file (*.onnx); default: the remembered / first model')
    if combined:
        ap.add_argument('--tool', choices=[s.key for s in specs], help='which tool to run / check (default for --check-updates and --download: all)')
    for s in specs:
        s.add_cli(ap)
    ap.add_argument('--check-updates', action='store_true', help='compare installed models and this script with the published manifest, then exit')
    ap.add_argument('--download', nargs='?', const='default', metavar='MODEL_ID|all', help='download a published model (default: the default model) into the models folder, then exit')
    ap.add_argument('--ui-test', action='store_true', help='open the window on a fake image, run Preview once, exit')
    ap.add_argument('--screenshots', help=argparse.SUPPRESS)
    a = ap.parse_args()
    chosen = [s for s in specs if not combined or not a.tool or s.key == a.tool]
    if a.check_updates or a.download:
        if combined:
            return cli_updates(chosen, a, lambda s: combined_upd(specs))
        return cli_updates(chosen, a)
    if a.ui_test:
        return run_app(specs, combined, None, fake_image(), smoke=True, shots=a.screenshots)
    if a.image:
        if not a.out:
            ap.error('--out is required with --image')
        spec = chosen[0] if not combined else (TOOLS.get(a.tool) if a.tool else None)
        if spec is None:
            ap.error('--tool is required with --image in the combined script')
        spec.check_cli(a, ap)
        run_file(spec, a)
        return 0
    try:
        import_engine(specs[0])   # import onnxruntime before Qt starts; any failure is kept and shown in the window
    except Exception:
        pass
    import sirilpy as s
    siril = s.SirilInterface()
    try:
        siril.connect()
        siril.cmd('requires', '1.4.0')
        return run_app(specs, combined, siril)
    finally:
        siril.disconnect()


# ============================================================================================ engines
def _engine_denoise():
    """Umbra Denoise engine: numpy + ONNX Runtime only (no Qt, no Siril, no torch).

    Same maths as the PyTorch pipeline (umbra/pipeline.py run_deployed): whole-image median / noise-sigma normalisation, asinh
    compression, 512 px tiles with a cosine overlap, regional noise guard, Detail split, mean fix, Pattern high-pass, then the
    model-independent saturated-star protection.

    params (all optional):
        denoise  0..1  strength of the noise removal                 (default 1.0)
        detail   0..1  gives back part of the removed fine detail    (default 0.0)
        pattern  0..1  strength of the structured-pattern removal    (default 0.0)
    data: float32 linear image, (H, W) mono or (C, H, W) with C in {1, 3}; NaN is kept as NaN.
    Returns (out, info), info = {'notes': [str], 'saturated': int}.

    Images smaller than one tile: PyTorch runs the network at the image's own size (a multiple of 32, e.g. 320 x 320 for
    300 x 250).  The network has global channel attention, so it depends on the tile size; the ONNX tile is fixed at 512, so
    the small padded tile is extended to 512 by mirror-tiling (period 2T), which keeps its global statistics (see PARITY_denoise.md).
    """
    import math
    import numpy as np

    TOOL = 'denoise'
    TILE = 512
    OVERLAP = 64
    K_COMP = 8.0
    GUARD_K = 4.0
    MEAN_FIX = 16.0
    COND_CH = 4
    NOTHING_TO_DO = 'Nothing to do: both sliders are at 0. The image was not changed.'


    def probe_feed():
        """Random inputs matching the ONNX inputs, for the GPU-vs-CPU probe."""
        rng = np.random.default_rng(1)
        return {'x': (rng.standard_normal((1, 3, TILE, TILE)) * 2).astype(np.float32),
                'c': np.zeros(1, np.float32),
                'cond': (rng.standard_normal((1, COND_CH)) * 0.5).astype(np.float32)}


    def _check(cancel):
        if cancel is not None and cancel():
            raise RuntimeError('Cancelled')




    def lmedian(a, axis=None):
        """torch.median semantics: the LOWER of the two middle values for an even count."""
        a = np.asarray(a)
        if axis is None:
            a = a.reshape(-1); axis = 0
        n = a.shape[axis]
        k = (n - 1) // 2
        return np.take(np.partition(a, k, axis=axis), k, axis=axis)


    def _resize_matrix(n_out, n_in):
        """Bilinear, align_corners=False (torch semantics), as an (n_out, n_in) matrix."""
        src = np.maximum((np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5, 0.0)
        i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        w = (src - i0).astype(np.float32)
        M = np.zeros((n_out, n_in), np.float32)
        M[np.arange(n_out), i0] += 1 - w
        M[np.arange(n_out), i1] += w
        return M


    def resize_bilinear(a, H, W):
        Ay = _resize_matrix(H, a.shape[1]); Ax = _resize_matrix(W, a.shape[2])
        return np.einsum('Hh,chw,Ww->cHW', Ay, a.astype(np.float32), Ax, optimize=True).astype(np.float32)


    def _sub(x, max_px=4_000_000):
        st = max(1, int((x.shape[-1] * x.shape[-2] / max_px) ** 0.5))
        return x[:, ::st, ::st]


    def haar_sigma(x):
        x = np.asarray(x, np.float32)
        x = x[:, : x.shape[1] // 2 * 2, : x.shape[2] // 2 * 2]
        a, b, c, d_ = x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]
        d = (a - b - c + d_) / 2.0
        ok = np.isfinite(a) & np.isfinite(d_) & (a != 0) & np.isfinite(d)
        out = []
        for ch in range(x.shape[0]):
            v = _sub(d[ch:ch + 1])[_sub(ok[ch:ch + 1])]
            out.append(np.float32(1.4826) * lmedian(np.abs(v - lmedian(v))) if v.size > 100 else np.float32(0.0))
        return np.array(out, np.float32)


    def image_stats(x):
        x = np.asarray(x, np.float32)
        xs = _sub(x)
        m = np.array([lmedian(np.nan_to_num(xs[c][np.isfinite(xs[c])])) for c in range(x.shape[0])], np.float32)
        s = haar_sigma(x)
        rng = np.array([np.quantile(np.nan_to_num(xs[c]).ravel(), 0.999) if xs[c].size < 16_000_000 else xs[c].max()
                        for c in range(x.shape[0])], np.float32) - m
        return m, np.maximum(s, 1e-5 * np.clip(np.abs(rng), 1e-12, None)).astype(np.float32)


    def local_sigma(x, block=32):
        """(C,H,W) -> local fine-scale noise sigma map (diagonal-Haar MAD per 32 px block, bilinear upsampled)."""
        C, H, W = x.shape
        x = x[:, : H // 2 * 2, : W // 2 * 2].astype(np.float32)
        d = np.abs(x[:, 0::2, 0::2] - x[:, 1::2, 0::2] - x[:, 0::2, 1::2] + x[:, 1::2, 1::2]) / 2.0
        b = block // 2
        h2, w2 = d.shape[1] // b * b, d.shape[2] // b * b
        if h2 == 0 or w2 == 0:
            return np.broadcast_to(haar_sigma(x)[:, None, None], (C, H, W)).astype(np.float32)
        t = np.nan_to_num(d[:, :h2, :w2]).reshape(C, h2 // b, b, w2 // b, b).transpose(0, 1, 3, 2, 4).reshape(C, h2 // b, w2 // b, b * b)
        med = np.float32(1.4826) * lmedian(t, axis=-1)
        return resize_bilinear(med, H, W)


    def _robust_sigma(d):
        out = []
        for c in range(d.shape[0]):
            v = _sub(d[c:c + 1]).ravel()
            v = v[np.isfinite(v)]
            out.append(np.float32(1.4826) * lmedian(np.abs(v - lmedian(v))) if v.size > 100 else np.float32(1.0))
        return np.maximum(np.array(out, np.float32), 1e-12)


    def noise_descriptor(x):
        """Noise description of the whole image for v3+ models (log binned/1x sigma at 2x and 4x, row/column anisotropy, 0)."""
        x = np.nan_to_num(np.asarray(x, np.float32))
        C, H, W = x.shape

        def det(z):
            z = z[:, : z.shape[1] // 2 * 2, : z.shape[2] // 2 * 2]
            a, b, c, d = z[:, 0::2, 0::2], z[:, 0::2, 1::2], z[:, 1::2, 0::2], z[:, 1::2, 1::2]
            return (a - b - c + d) / 2.0, (a + b - c - d) / 2.0, (a - b + c - d) / 2.0

        def pool(z, f):
            z = z[:, : z.shape[1] // f * f, : z.shape[2] // f * f]
            return z.reshape(C, z.shape[1] // f, f, z.shape[2] // f, f).mean((2, 4))

        dg, dh, dv = det(x)
        s1 = _robust_sigma(dg)
        s2 = 2.0 * _robust_sigma(det(pool(x, 2))[0])
        s4 = 4.0 * _robust_sigma(det(pool(x, 4))[0])
        v = np.array([np.log(s2 / s1).mean(), np.log(s4 / s1).mean(), np.log(_robust_sigma(dh) / _robust_sigma(dv)).mean(), 0.0], np.float32)
        return np.clip(v, -3.0, 3.0)


    def _blur(x, sigma):
        """Separable Gaussian (radius int(3 sigma)), reflect padding (replicate on tiny images); channels run in parallel."""
        from scipy.ndimage import correlate1d
        from concurrent.futures import ThreadPoolExecutor
        r = max(1, int(3 * sigma))
        t = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-0.5 * (t / sigma) ** 2); k = (k / k.sum()).astype(np.float32)
        mode = 'reflect' if min(x.shape[-2:]) > r else 'edge'

        def one(c):
            y = np.pad(c, ((0, 0), (r, r)), mode=mode)
            y = correlate1d(y, k, axis=1, mode='constant')[:, r:-r]
            y = np.pad(y, ((r, r), (0, 0)), mode=mode)
            return correlate1d(y, k, axis=0, mode='constant')[r:-r]
        with ThreadPoolExecutor(max_workers=min(len(x), 4)) as ex:
            return np.stack(list(ex.map(one, list(x)))).astype(np.float32)


    def _window(n, ov):
        w = np.ones(n, np.float32)
        if ov > 0:
            r = np.linspace(0, 1, ov + 2, dtype=np.float32)[1:-1]
            r = (0.5 - 0.5 * np.cos(np.pi * r)).astype(np.float32)
            w[:ov] = r; w[-ov:] = r[::-1]
        return w


    def _mirror_to(a, n):
        """(C, T, T) -> (C, n, n), n >= T, by mirror-tiling with period 2T (symmetric: ... c b a | a b c ... c b a | a b c ...)."""
        T = a.shape[-1]
        idx = np.arange(n) % (2 * T)
        idx = np.where(idx >= T, 2 * T - 1 - idx, idx)
        return np.ascontiguousarray(a[:, idx][:, :, idx])


    def residuals(net, xn, want_pattern, cond=None, progress=None, cancel=None):
        """xn (3,H,W) normalised -> r0 (and r1 = pattern-removing residual when want_pattern), sigma units.
        net(x, c, cond) runs the fixed 512 tile.  Same tiling as umbra.infer.residuals."""
        C, H, W = xn.shape
        nout = 2 if want_pattern else 1
        if max(H, W) < TILE:
            # PyTorch tile for a small image: T = ceil(max/32)*32 (min 32), one tile, padded to T x T (reflect, else replicate)
            Tt = min(TILE, max(32, int(math.ceil(max(H, W) / 32)) * 32))
            mode = 'reflect' if (Tt - H < H and Tt - W < W) else 'edge'
            xc = K_COMP * np.arcsinh(np.pad(xn, ((0, 0), (0, Tt - H), (0, Tt - W)), mode=mode) / K_COMP)
            t = _mirror_to(xc, TILE)[None] if Tt < TILE else np.ascontiguousarray(xc[None], np.float32)
            out = []
            for j in range(nout):
                _check(cancel)
                out.append(net(t.astype(np.float32, copy=False), float(j), cond)[0][:, :H, :W])
            if progress:
                progress(1, 1)
            return out[0], (out[1] if want_pattern else None)
        T, st = TILE, TILE - OVERLAP
        ny = max(1, math.ceil((H - OVERLAP) / st)) if H > T else 1
        nx = max(1, math.ceil((W - OVERLAP) / st)) if W > T else 1
        Hp, Wp = max(H, (ny - 1) * st + T), max(W, (nx - 1) * st + T)
        mode = 'reflect' if (Hp - H < H and Wp - W < W) else 'edge'
        xc = K_COMP * np.arcsinh(np.pad(xn, ((0, 0), (0, Hp - H), (0, Wp - W)), mode=mode) / K_COMP)
        acc = np.zeros((nout, C, Hp, Wp), np.float32); wsum = np.zeros((Hp, Wp), np.float32)
        w2 = _window(T, OVERLAP if ny > 1 else 0)[:, None] * _window(T, OVERLAP if nx > 1 else 0)[None, :]
        coords = [(iy * st, ix * st) for iy in range(ny) for ix in range(nx)]
        for n, (y0, x0) in enumerate(coords, 1):
            _check(cancel)
            t = np.ascontiguousarray(xc[None, :, y0:y0 + T, x0:x0 + T], np.float32)
            for j in range(nout):
                acc[j, :, y0:y0 + T, x0:x0 + T] += net(t, float(j), cond)[0] * w2
            wsum[y0:y0 + T, x0:x0 + T] += w2
            if progress:
                progress(n, len(coords))
        r = (acc / np.maximum(wsum, 1e-8))[:, :, :H, :W]
        return r[0], (r[1] if want_pattern else None)


    def denoise_chw(net, y, d=1.0, p=0.0, detail=0.0, progress=None, cancel=None):
        Y = np.ascontiguousarray(y, np.float32)
        fin = np.isfinite(Y)
        Y0 = np.nan_to_num(Y)
        m, s = image_stats(Y0)
        m, s = m[:, None, None], s[:, None, None]
        xn = (Y0 - m) / s
        cond = noise_descriptor(xn)[None]
        r0, r1 = residuals(net, xn, p > 0, cond, progress, cancel)
        _check(cancel)
        ls = local_sigma(Y0) / s                                                     # regional noise guard
        rms = np.sqrt(np.clip(_blur(r0 * r0, 16.0), 0, None))
        g = np.minimum(GUARD_K * ls / np.maximum(rms, 1e-12), 1.0)
        g = _blur(g, 8.0)
        r0g = r0 * g
        if detail and detail > 0:                                                    # Detail: give back part of the removed fine / mid residual
            D = float(min(max(detail, 0.0), 1.0))
            b1 = _blur(r0g, 1.2); b16 = _blur(r0g, 16.0)
            r0g = (r0g - b1) * (1.0 - 0.6 * D) + (b1 - b16) * (1.0 - D) + b16
        r0g = r0g - _blur(r0g, MEAN_FIX)                                             # never move levels / colours at >= ~32 px
        out = xn + d * r0g
        if r1 is not None:
            pat = r1 - r0
            pat = pat - _blur(pat, 32.0)
            out = out + p * pat
        out = out * s + m
        return np.where(fin, out, Y).astype(np.float32)


    def denoise_array(net, data, d=1.0, p=0.0, detail=0.0, progress=None, cancel=None):
        """2D mono or (C,H,W), C in {1, 3}. Mono is replicated to 3 channels and averaged back."""
        a = np.asarray(data, np.float32)
        if a.ndim == 2:
            cube = a[None]
        elif a.ndim == 3 and a.shape[0] in (1, 3):
            cube = a
        else:
            raise ValueError(f'unsupported shape {a.shape}')
        if cube.shape[0] == 3:
            out = denoise_chw(net, cube, d, p, detail, progress, cancel)
        else:
            out = denoise_chw(net, np.repeat(cube, 3, 0), d, p, detail, progress, cancel).mean(0, keepdims=True)
        return out[0] if a.ndim == 2 else out


    # ----------------------------------------------------------------------------------------------- saturated-star protection (verbatim from the Siril script)
    TIGHTEN = 0.6          # luminance never drops below background + 0.6 x (input - background) inside a protected star
    COLOR_SNR = 8.0        # input colour is used as is above this many noise sigmas over background, smoothed below it
    FOOT_FRAC = 0.12       # footprint = input luminance above background + 12 % of the star's peak above background
    _INT_MAX = (255.0, 1023.0, 4095.0, 16383.0, 65535.0)


    def _looks_clipped(c, m):
        """True when channel max `m` looks like a clipping level: integer-format maximum, or shared by >= 3 pixels."""
        if m in _INT_MAX:
            return True
        return int(np.count_nonzero(c >= m)) >= 3


    def _plateau_cores(c, hi=0.8, flat=0.99, min_px=3):
        """Flat-topped star cores at ANY level (float stacks rescaled so the clip level is below the channel max, often different per
        channel): connected regions above hi x channel max whose pixels within (1 - flat) of the region's own peak number >= min_px.
        A real (unclipped) star peak has only 1-2 pixels that close to its maximum."""
        from scipy import ndimage as ndi
        m = float(c.max())
        if not np.isfinite(m) or m <= 0:
            return None
        high = c >= hi * m
        if not high.any() or high.sum() > c.size // 20:
            return None
        lab, n = ndi.label(high, structure=np.ones((3, 3), bool))
        idx = np.flatnonzero(high)                                  # work on the few bright pixels only (ndi.maximum over the full frame took ~1.7 s)
        ll, vv = lab.ravel()[idx], c.ravel()[idx]
        pk = np.full(n + 1, -np.inf, np.float32)
        np.maximum.at(pk, ll, vv)
        nr = vv >= flat * pk[ll]
        cnt = np.bincount(ll[nr], minlength=n + 1)
        keep = cnt >= min_px
        keep[0] = False
        sel = idx[nr & keep[ll]]
        if sel.size == 0:
            return None
        out = np.zeros(c.shape, bool)
        out.ravel()[sel] = True
        return out


    def find_saturated_cores(data, frac=0.97):
        """Core mask (H, W) of saturated / near-saturated stars in a HxW or CxHxW image, or None.
        Per channel (union across channels, single-channel clips included): pixels >= frac * channel max, when the image looks clipped
        (integer-format max or max shared by >= 3 pixels) and the pixel is star-like (>= 2 of its 8 neighbours above half its value, or
        1 above half and 3 above 10%, or the other channels above 10% there, or >= 3 neighbours of the luminance above 30% of the
        luminance there; hot pixels have a dark neighbourhood and are rejected)."""
        planes = data[None] if data.ndim == 2 else data
        lumi = planes.mean(0) if len(planes) > 1 else planes[0]
        core = None
        for c in planes:                          # flat-topped cores below the channel max (rescaled float stacks)
            pc = _plateau_cores(c)
            if pc is not None:
                core = pc if core is None else (core | pc)
        for k, c in enumerate(planes):
            m = float(c.max())
            if not np.isfinite(m) or m <= 0 or not _looks_clipped(c, m):
                continue
            ys, xs = np.nonzero(c >= m * frac)
            if ys.size == 0 or ys.size > c.size // 20:
                continue
            H, W = c.shape
            v = c[ys, xs]
            lv = lumi[ys, xs]
            n50 = np.zeros(ys.size, np.int32)
            n10 = np.zeros(ys.size, np.int32)
            nl30 = np.zeros(ys.size, np.int32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = ys + dy, xs + dx
                    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                    nb = np.zeros(ys.size, np.float32)
                    nb[ok] = c[yy[ok], xx[ok]]
                    n50 += (nb > 0.5 * v)
                    n10 += (nb > 0.1 * v)
                    if len(planes) > 1:
                        nl = np.zeros(ys.size, np.float32)
                        nl[ok] = lumi[yy[ok], xx[ok]]
                        nl30 += (nl > 0.3 * lv)
            keep = (n50 >= 2) | ((n50 >= 1) & (n10 >= 3))       # star-like: hot pixels have a dark neighbourhood
            if len(planes) > 1:                                   # or the other channels / the luminance light up at the same pixel (a real point source)
                oth = (sum(p[ys, xs] for j, p in enumerate(planes) if j != k) / (len(planes) - 1))
                keep |= (oth > 0.1 * v) | (nl30 >= 3)
            if not keep.any():
                continue
            cc = np.zeros((H, W), bool)
            cc[ys[keep], xs[keep]] = True
            core = cc if core is None else (core | cc)
        return core


    def protect_saturated(inp, out):
        """Saturated-star protection v3 (model independent): returns (result, n_stars).
        Each clipped star (>= 1 clipped pixel in any channel) gets a footprint from the INPUT luminance (above background + 12 % of the
        star's peak, grown 1-2 px, radius capped at 25 px; at least as large as the v2 ring), feathered at its edge. Inside it the colour
        (channel / mean ratio) follows the input, the whole core included; clipped pixels (small stars; large ones also their 1 px rim) are the input exactly. Luminance keeps
        the model's change but never drops below background + 0.6 x (input - background) and never exceeds max(input, smoothed model
        luminance). No stars found: `out` is returned untouched."""
        from scipy import ndimage as ndi
        core = find_saturated_cores(inp)
        if core is None or not core.any():
            return out, 0
        inp = np.asarray(inp, np.float32)
        out = np.asarray(out, np.float32)
        box = np.ones((3, 3), bool)
        dil = ndi.binary_dilation(core, structure=box)
        lab, n = ndi.label(dil, structure=box)
        area = np.bincount(lab[dil], minlength=n + 1).astype(np.float32)
        rad = np.sqrt(np.maximum(area, 1.0) / np.pi)
        H, W = dil.shape
        mono = inp.ndim == 2
        mi = inp if mono else inp.mean(0)
        mo = out if mono else out.mean(0)
        wc = np.zeros((H, W), np.float32)         # protection weight (1 = input colour); also the footprint feather
        wl = np.ones((H, W), np.float32)          # luminance weight (1 = model luminance); ramps up over 2 px outside the core
        bgm = np.zeros((H, W), np.float32)        # local background per pixel (from the star whose weight is largest)
        msm = np.empty_like(mo)                   # smoothed model luminance (ceiling), filled inside the windows
        floor_e = 1e-6 * max(float(mi.max()), 1.0)
        rsm = None if mono else np.empty_like(inp)  # smoothed input colour ratio (channel / mean), filled inside the windows
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            r_i = float(rad[i])
            R_ring = min(20.0, 3.0 + 2.0 * r_i)                      # v2 ring radius (kept as a minimum extent)
            R_cap = min(25.0, 6.0 + 3.0 * r_i)                       # footprint radius cap
            F = 2.0 + 0.1 * r_i                                      # footprint edge feather
            m = int(np.ceil(max(R_cap + F, R_ring))) + 6
            y0, y1 = max(sl[0].start - m, 0), min(sl[0].stop + m, H)
            x0, x1 = max(sl[1].start - m, 0), min(sl[1].stop + m, W)
            lc = lab[y0:y1, x0:x1]
            mc = lc == i
            d = ndi.distance_transform_edt(~mc).astype(np.float32)
            mw = mi[y0:y1, x0:x1]
            ann = (d > R_cap) & (d <= R_cap + 4.0) & (lc == 0)
            bg = float(np.percentile(mw[ann], 25)) if ann.sum() >= 8 else float(np.percentile(mw, 25))
            peak = float(mw[mc].max())
            thr = bg + FOOT_FRAC * max(peak - bg, 0.0)
            cand = (ndi.gaussian_filter(mw, 0.7) > thr) & (d <= R_cap)
            cl, _ = ndi.label(cand | mc, structure=box)
            fp = np.isin(cl, np.unique(cl[mc]))
            fp = ndi.binary_fill_holes(fp)
            g = 5 if r_i > 3.0 else 3
            fp = ndi.maximum_filter(fp.view(np.uint8), size=g).astype(bool) | mc      # grown 1 px (2 for large stars)
            dfp = ndi.distance_transform_edt(~fp).astype(np.float32)
            t = np.clip(dfp / F, 0, 1)
            w = 1 - t * t * (3 - 2 * t)                              # plateau 1 inside the footprint, smooth edge
            tr = np.clip(d / R_ring, 0, 1)
            w = np.maximum(w, 1 - tr * tr * (3 - 2 * tr))            # never less than the v2 ring
            dl = d if r_i > 3.0 else ndi.distance_transform_edt(~(mc & core[y0:y1, x0:x1])).astype(np.float32)   # luminance eases in from the clipped pixels
            t2 = np.clip(dl / 2.0, 0, 1)
            l = t2 * t2 * (3 - 2 * t2)
            wcs = wc[y0:y1, x0:x1]
            upd = w > wcs
            bgm[y0:y1, x0:x1] = np.where(upd, bg, bgm[y0:y1, x0:x1])
            msm[y0:y1, x0:x1] = np.where(upd, ndi.gaussian_filter(mo[y0:y1, x0:x1], 1.0), msm[y0:y1, x0:x1])
            if not mono:        # input colour ratio: the pixel's own where it is bright (high S/N), a flux-weighted smoothed one in the faint halo (no pixel-noise speckle)
                sg = 1.0 + 0.15 * r_i
                iw = inp[:, y0:y1, x0:x1]
                num = ndi.gaussian_filter(iw, (0, sg, sg))
                den = ndi.gaussian_filter(mw, sg)
                rs = np.where(den > floor_e, num / np.maximum(den, floor_e), 1.0)
                ri_w = np.where(mw > floor_e, iw / np.maximum(mw, floor_e), 1.0)
                sig = 1.4826 * float(np.median(np.abs(mw[ann] - np.median(mw[ann])))) if ann.sum() >= 8 else 0.0
                q = np.clip((mw - bg) / max(COLOR_SNR * sig, floor_e), 0, 1)
                q = q * q * (3 - 2 * q)
                rs = ri_w * q + rs * (1 - q)
                rsm[:, y0:y1, x0:x1] = np.where(upd[None], rs, rsm[:, y0:y1, x0:x1])
            wc[y0:y1, x0:x1] = np.maximum(wcs, w)
            wl[y0:y1, x0:x1] = np.minimum(wl[y0:y1, x0:x1], l)
        sel = np.nonzero(wc > 0)                          # everything below touches only these pixels (the protected footprints)
        hard = core[sel] | (dil[sel] & (area[lab[sel]] > 30))     # exactly the input: clipped pixels (small stars), plus the 1 px rim of large ones
        w_ = wc[sel]
        wf = np.clip(w_ * 1.5, 0, 1)
        mi_, mo_, bg_ = mi[sel], mo[sel], bgm[sel]
        lum = mi_ + (mo_ - mi_) * wl[sel]                     # model luminance, eased in from the input at the core edge
        lum = lum + (np.minimum(lum, np.maximum(mi_, msm[sel])) - lum) * wf       # no hot cores: not brighter than input / smoothed model
        floor = np.where(mi_ > bg_, bg_ + TIGHTEN * (mi_ - bg_), mi_)    # at most (1 - TIGHTEN) of the input's excess over background is removed
        lum = lum + (np.maximum(lum, floor) - lum) * wf
        res = out.copy()
        if mono:
            res[sel] = np.where(hard, inp[sel], lum)
        else:
            rm = np.where(mo_ > floor_e, out[(slice(None),) + sel] / np.maximum(mo_, floor_e), 1.0)    # colour = channel / mean (ratio), so it scales with the clamped luminance
            ratio = np.clip(rm + (rsm[(slice(None),) + sel] - rm) * w_, 0.0, 3.0)    # colour follows the input inside the footprint
            r_ = lum[None] * ratio
            r_ = np.minimum(r_, inp.reshape(len(inp), -1).max(1)[:, None])    # never above the input's clip level
            res[(slice(None),) + sel] = np.where(hard[None], inp[(slice(None),) + sel], r_)
        return res.astype(np.float32, copy=False), int(n)




    # ----------------------------------------------------------------------------------------------- Engine
    class Engine:
        provider = 'CPU'
        note = None

        def __init__(self, model_path, session_factory):
            self.model_path = str(model_path)
            self._run, self.provider, self.note = session_factory(self.model_path, probe_feed())

        def _net(self, x, c, cond=None):
            feed = {'x': x, 'c': np.array([c], np.float32),
                    'cond': np.asarray(cond if cond is not None else np.zeros((1, COND_CH)), np.float32)}
            return self._run(feed)[0]

        def run(self, data, params=None, progress=None, cancel=None):
            prm = dict(params or {})
            d = float(min(max(prm.get('denoise', 1.0), 0.0), 1.0))
            D = float(min(max(prm.get('detail', 0.0), 0.0), 1.0))
            p = float(min(max(prm.get('pattern', 0.0), 0.0), 1.0))
            data = np.ascontiguousarray(data, np.float32)
            if d == 0 and p == 0:
                return data.copy(), {'notes': [NOTHING_TO_DO], 'saturated': 0}
            out = denoise_array(self._net, data, d, p, D, progress, cancel)
            _check(cancel)
            out, n = protect_saturated(data, np.asarray(out, np.float32))        # always on, model independent
            notes = [f'Protected {n} saturated stars.'] if n else []
            return out, {'notes': notes, 'saturated': int(n)}
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


ENGINES['denoise'] = _engine_denoise()

# ============================================================================================ tools
def _tool_denoise():
    # ---- Umbra Denoise: tool layer ----
    DENOISE_TIP = 'Strength of the noise removal. 100% = full noise removal, 0 = off (Pattern noise can still run on its own).'
    DETAIL_TIP = ('Keeps part of what Denoise removed, split by scale: faint mottled structure (nebula texture, outer halos) comes back '
                  'together with a little grain. 0 = full noise removal. 25-50 suits faint, low-contrast targets.')
    PATTERN_TIP = ('For PATTERN noise only: blotchy mottle repeated in every sub (calibration / no dithering). Random noise is removed by Denoise. '
                   'Pattern cannot be told apart from real faint signal with certainty, so it may remove some. Leave at 0 unless you see blotches. '
                   'Works on its own, without Denoise.')


    class DenoisePanel:
        def __init__(self, ui, layout, host):
            pct = lambda v: f'{v}%'
            self.denoise = ui.slider(layout, 'Denoise', 0, 100, 100, pct, DENOISE_TIP)
            self.detail = ui.slider(layout, 'Detail', 0, 100, 0, pct, DETAIL_TIP)
            self.pattern = ui.slider(layout, 'Pattern noise', 0, 100, 0, pct, PATTERN_TIP)

        def model_changed(self, path):
            pass

        def params(self):
            return {'denoise': self.denoise.value() / 100.0, 'detail': self.detail.value() / 100.0, 'pattern': self.pattern.value() / 100.0}

        def settings(self):
            return dict(denoise=self.denoise.value(), detail=self.detail.value(), pattern=self.pattern.value())

        def set_settings(self, d):
            for k in ('denoise', 'detail', 'pattern'):
                if isinstance(d.get(k), int):
                    getattr(self, k).setValue(d[k])


    class DenoiseSpec(ToolSpec):
        def build_panel(self, ui, layout, host):
            return DenoisePanel(ui, layout, host)

        def add_cli(self, ap):
            ap.add_argument('--denoise', type=float, default=1.0, help='denoise: noise removal 0..1 (default 1)')
            ap.add_argument('--detail', type=float, default=0.0, help='denoise: gives back fine detail 0..1 (default 0)')
            ap.add_argument('--pattern', type=float, default=0.0, help='denoise: pattern-noise removal 0..1 (default 0)')

        def check_cli(self, a, ap):
            if not (0 <= a.denoise <= 1 and 0 <= a.detail <= 1 and 0 <= a.pattern <= 1):
                ap.error('--denoise, --detail and --pattern must lie in [0, 1]')

        def params_from_args(self, a):
            return {'denoise': a.denoise, 'detail': a.detail, 'pattern': a.pattern}

        def label(self, p):
            return f"d {float(p.get('denoise', 1)):.2f} detail {float(p.get('detail', 0)):.2f} pattern {float(p.get('pattern', 0)):.2f}"


    SPEC = DenoiseSpec(
        key='denoise', name='Umbra Denoise', script='Umbra_Denoise', repo=f'{GITHUB_USER}/umbra-denoise',
        prefix='UMBRA_DENOISE_', folder='Umbra Denoise',
        labels={'umbra_denoise_v3.1_experimental': 'Umbra Denoise v3.1 (experimental)'},
        default_stems=('umbra_denoise_v3.1_experimental',),
        notes=['AI denoise for linear images.'])
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


TOOLS['denoise'] = _tool_denoise().SPEC


if __name__ == '__main__':
    raise SystemExit(main([TOOLS[k] for k in ('denoise',)], combined=False, doc=__doc__))
