from mcp.server.fastmcp import FastMCP

# Creates an MCP server and gives it a name.
mcp = FastMCP("AI Code Review Tools")


# Registers this Python function as an MCP tool.
@mcp.tool()
def calculate_total_changes(
    additions: int,
    deletions: int
) -> int:

    return additions + deletions


# Starts the MCP server when this file is executed directly.
if __name__ == "__main__":
    mcp.run(transport="stdio")