import flwr as fl

def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    acc  = sum(n * m["accuracy"]  for n, m in metrics) / total
    prec = sum(n * m["precision"] for n, m in metrics) / total
    rec  = sum(n * m["recall"]    for n, m in metrics) / total
    f1   = sum(n * m["f1"]        for n, m in metrics) / total
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
