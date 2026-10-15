
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Calls our MCP calculator server and returns the result.
async def calculate_changes_via_mcp(
    additions: int,
    deletions: int
) -> int:

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "calculate_total_changes",
                arguments={
                    "additions": additions,
                    "deletions": deletions
                }
            )
            print("MCP calculator returned:", result.content)

            if result.isError:
                raise RuntimeError(
                    f"MCP tool failed: {result.content}"
                )

            return int(result.content[0].text)
