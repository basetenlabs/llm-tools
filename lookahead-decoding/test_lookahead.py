"""Exercise the actual fetch function without plotting or GPU dependencies."""
import ast
import asyncio
import math
from pathlib import Path
import unittest
import subprocess
import sys

source = ast.parse(Path(__file__).with_name("lookahead.py").read_text())
fetch_node = next(node for node in source.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "fetch")

class Clock:
    now = 1.0
    def time(self):
        return self.now

class Response:
    def __init__(self, clock, fail=False):
        self.clock, self.fail, self.read_called, self.status_checked = clock, fail, False, False
    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass
    def raise_for_status(self):
        self.status_checked = True
        if self.fail: raise RuntimeError("HTTP 500")
    async def read(self):
        self.read_called = True
        self.clock.now += 2
        return b'{"output":"done"}'

class Session:
    def __init__(self,response): self.response=response
    def post(self,*args,**kwargs): return self.response

class FetchTests(unittest.TestCase):
    def fetch(self, response, clock):
        namespace = {"time": clock}
        exec(compile(ast.Module(body=[fetch_node], type_ignores=[]), "lookahead.py", "exec"), namespace)
        return asyncio.run(namespace["fetch"](Session(response), "https://example.invalid", {}, {}))
    def test_times_complete_body(self):
        clock=Clock(); response=Response(clock)
        self.assertEqual(self.fetch(response,clock),2)
        self.assertTrue(response.read_called)
        self.assertTrue(response.status_checked)
    def test_parameter_ranges_include_build_maximum(self):
        main_node=next(n for n in source.body if isinstance(n,ast.AsyncFunctionDef) and n.name=="main")
        assigns=[n for n in main_node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"window_sizes","verification_set_sizes","ngram_sizes"} for t in n.targets)]
        append_node=next(n for n in main_node.body if isinstance(n,ast.If) and "ngram_sizes" in ast.unparse(n.test))
        for maximum in [1,3,5,64]:
            namespace={"math":math,"engine_window_size":maximum,"engine_verification_set":maximum,"engine_ngram_size":maximum}
            exec(compile(ast.Module(body=assigns+[append_node],type_ignores=[]),"lookahead.py","exec"),namespace)
            for name in ["window_sizes","verification_set_sizes","ngram_sizes"]:
                values=namespace[name]
                self.assertIn(maximum,values)
                self.assertTrue(all(1<=v<=maximum for v in values))

    def test_invalid_grid_is_rejected_before_runtime_imports(self):
        result=subprocess.run([sys.executable,str(Path(__file__).with_name("lookahead.py")),"--model","test","--api_key","test-key","--window_size","1","--ngram_size","0","--verification_size","1"],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn("must be at least 1",result.stderr)

    def test_http_error_is_not_a_fast_success(self):
        clock=Clock(); response=Response(clock,fail=True)
        with self.assertRaisesRegex(RuntimeError,"HTTP 500"): self.fetch(response,clock)
        self.assertFalse(response.read_called)

if __name__ == "__main__": unittest.main()
