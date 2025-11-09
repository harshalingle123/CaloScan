# mcp/server.py
from typing import Callable, Dict

class Server:
    """
    A minimal stub to mimic 'mcp.server.Server' so you can use
    @server.tool() and server.run().
    """
    def __init__(self, name: str):
        self.name = name
        self._tools: Dict[str, Callable] = {}

    def tool(self):
        """Decorator to register a function as a 'tool'."""
        def decorator(func: Callable):
            self._tools[func.__name__] = func
            return func
        return decorator

    def run(self, **kwargs):
        """
        Placeholder run method.
        Currently just prints available tools.
        """
        print(f"🚀 Server '{self.name}' running (stub).")
        print("Registered tools:")
        for name in self._tools:
            print(f"  - {name}")
