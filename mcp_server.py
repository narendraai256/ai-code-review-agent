from mcp.server.fastmcp import FastMCP
from tools import calculate_total_changes as calculate_total_changes_local

# Creates an MCP server and gives it a name.
mcp = FastMCP("AI Code Review Tools")


# Registers this Python function as an MCP tool.
@mcp.tool()
def calculate_total_changes(
    additions: int,
    deletions: int
) -> int:

    return calculate_total_changes_local(additions, deletions)


# Starts the MCP server when this file is executed directly.
if __name__ == "__main__":
    mcp.run(transport="stdio")
