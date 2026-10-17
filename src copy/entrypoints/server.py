from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents

def server_fn(context: Context) -> ServerAppComponents:
    if str(context.run_config.get("mode", "deploy")).lower() == "sim":
        from src.entrypoints.sim_server import server_fn as sim_server_fn
        return sim_server_fn(context)
    else:
        from src.entrypoints.deploy_server import server_fn as deploy_server_fn
        return deploy_server_fn(context)

app = ServerApp(server_fn=server_fn)