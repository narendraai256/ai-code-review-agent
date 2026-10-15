
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Defines how the client should start our MCP server.
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"]
)


async def main():

    # Starts the MCP server and opens communication with it.
    async with stdio_client(server_params) as (read, write):

        # Creates a communication session with the MCP server.
        async with ClientSession(read, write) as session:

            # Establishes the MCP connection.
            await session.initialize()

            # Requests the list of available MCP tools.
            tools = await session.list_tools()

            print("Available MCP tools:")
            for tool in tools.tools:
                print(tool.name)

            # Calls our calculator through MCP.
            result = await session.call_tool(
                "calculate_total_changes",
                arguments={
                    "additions": 100,
                    "deletions": 50
                }
            )

            print("MCP tool result:")
            print(result.content)


# Runs our asynchronous client program.
if __name__ == "__main__":
    asyncio.run(main())
