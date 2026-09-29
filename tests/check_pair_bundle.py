"""Compare bundled application code/assets to reviewed source without executing it."""
import sys
import types
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader


def signature(code):
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
            tuple(signature(c) if isinstance(c, types.CodeType) else c for c in code.co_consts))


def main():
    root=Path(__file__).resolve().parents[1]
    artifact=Path(sys.argv[1])
    if artifact.suffix=='.pyz':
        archive=None
        pyz=ZlibArchiveReader(str(artifact))
    else:
        archive=CArchiveReader(str(artifact))
        pyz=archive.open_embedded_archive('PYZ.pyz')
    modules = ['pair_signals','pair_portfolio','pair_market_service','pair_web']
    if '--dual' in sys.argv:
        modules += ['boll_short_signals','boll_short_portfolio','boll_short_service']
    for module in modules:
        source=root/'futures_strategy'/f'{module}.py'
        expected=compile(source.read_text(encoding='utf-8'),str(source),'exec',dont_inherit=True)
        actual=pyz.extract('futures_strategy.'+module)
        if signature(actual)!=signature(expected):
            raise SystemExit(f'STALE BUNDLE: {module}')
        print(f'CURRENT: {module}')
    if archive:
        for name in ['index.html','app.css','app.js']+(['boll.js'] if '--dual' in sys.argv else []):
            key=next(k for k in archive.toc if k.replace('\\','/')==f'futures_strategy/web/pairs/{name}')
            assert archive.extract(key)==(root/'futures_strategy'/'web'/'pairs'/name).read_bytes(), name
            print(f'CURRENT: {name}')


if __name__=='__main__':
    main()
