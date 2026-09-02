def batched(iterable, batch_size: int):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def iter_batches(records, batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]