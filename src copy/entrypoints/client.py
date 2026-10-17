from flwr.common import Context
from flwr.client import ClientApp

def client_fn(context: Context):
    if str(context.run_config.get("mode", "deploy")).lower() == "sim":
        from src.entrypoints.sim_client import client_fn as sim_client_fn
        return sim_client_fn(context)
    else:
        from src.entrypoints.deploy_client import client_fn as deploy_client_fn
        return deploy_client_fn(context)

app = ClientApp(client_fn=client_fn)