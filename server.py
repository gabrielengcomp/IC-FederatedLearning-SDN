import flwr as fl


def weighted_average(metrics):
    accs = [n * m["accuracy"] for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {"accuracy": sum(accs) / total}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,        # usa 100% dos clientes disponíveis
    fraction_evaluate=1.0,
    min_fit_clients=2,       # mínimo de 2 clientes por rodada
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- adicionar

)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)