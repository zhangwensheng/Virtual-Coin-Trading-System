"""Exercise the real startup import with the packaging exclusions enforced."""
import ast
import importlib.abc
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
tree=ast.parse((root/'PairDesk.spec').read_text(encoding='utf-8'))
excluded=next(ast.literal_eval(k.value) for n in ast.walk(tree) if isinstance(n,ast.Call)
              and isinstance(n.func,ast.Name) and n.func.id=='Analysis' for k in n.keywords if k.arg=='excludes')


class Exclusions(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if any(fullname==m or fullname.startswith(m+'.') for m in excluded):
            raise ModuleNotFoundError('EXCLUDED: '+fullname,name=fullname)


sys.meta_path.insert(0,Exclusions())
from futures_strategy.pair_web import create_server
print('Startup imports pass with bundle exclusions')
