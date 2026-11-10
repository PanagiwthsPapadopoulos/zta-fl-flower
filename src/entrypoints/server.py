from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents

def server_fn(context: Context) -> ServerAppComponents:
    if str(context.run_config.get("mode", "deploy")).lower() == "sim":
        from src.entrypoints.sim_server import server_fn as sim_server_fn
        return sim_server_fn(context)
    else:
        from src.entrypoints.deploy_server import server_fn as deploy_server_fn
        return deploy_server_fn(context)

class TrackedServerApp(ServerApp):
    def __call__(self, grid, context):
        if str(context.run_config.get("mode", "deploy")).lower() != "sim":
            return super().__call__(grid, context)
        from src.shared.utils.config_loader import get_merged_config
        from src.shared.utils.simulation_lifecycle import track_simulation
        with track_simulation(get_merged_config(context.run_config), context.run_config):
            return super().__call__(grid, context)


app = TrackedServerApp(server_fn=server_fn)
