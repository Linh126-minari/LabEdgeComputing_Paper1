import numpy as np

from src.env.entities import Task, User
from src.env.traffic import (
    TrafficConfig,
    TrafficType,
    TaskGenerator,
    MultiUserTrafficManager,
)


def test_task_generator_basic_task_properties():
    """TaskGenerator.generate_task should respect TrafficConfig bounds."""
    config = TrafficConfig(
        task_size_min=200_000.0,
        task_size_max=200_000.0,
        cycles_per_bit=1500.0,
        max_delay=2.0,
    )
    gen = TaskGenerator(config)

    np.random.seed(0)
    task = gen.generate_task()

    assert isinstance(task, Task)
    # Size should match the fixed min/max
    assert task.size == 200_000.0
    assert task.cycles_per_bit == 1500.0
    assert task.max_delay == 2.0
    # Statistics should be updated
    assert gen.total_tasks_generated == 1
    assert len(gen.task_sizes_generated) == 1


def test_task_generator_should_generate_poisson():
    """Poisson arrival in should_generate is controlled by np.random.random."""
    config = TrafficConfig(
        traffic_type=TrafficType.POISSON,
        arrival_rate=0.5,
    )
    gen = TaskGenerator(config)

    # Force np.random.random to return a value < prob so it always generates
    original_random = np.random.random
    try:
        np.random.random = lambda: 0.0  # always smaller than any positive prob
        assert bool(gen.should_generate(user_id=0, current_time=0.0, dt=1.0)) is True
    finally:
        np.random.random = original_random


def test_multi_user_traffic_manager_deterministic_arrivals():
    """MultiUserTrafficManager with deterministic traffic should generate one task per user on first step."""
    config = TrafficConfig(
        traffic_type=TrafficType.DETERMINISTIC,
        arrival_rate=1.0,  # one task per second
    )
    num_users = 2
    manager = MultiUserTrafficManager(num_users=num_users, config=config)

    users = [
        User(id=i, position=np.array([0.0, 0.0]), cpu_frequency=1e9)
        for i in range(num_users)
    ]

    new_tasks = manager.step(users, dt=0.1)

    # Should have generated one task for each user
    assert len(new_tasks) == num_users
    for i in range(num_users):
        assert i in new_tasks
        assert isinstance(new_tasks[i], Task)
        # Queues should each contain exactly one task
        assert manager.get_queue_length(i) == 1
        assert manager.tasks_generated_per_user[i] == 1


def test_multi_user_traffic_manager_pending_and_statistics():
    """Test queue operations and statistics reporting of MultiUserTrafficManager."""
    config = TrafficConfig(
        traffic_type=TrafficType.DETERMINISTIC,
        arrival_rate=1.0,
    )
    num_users = 1
    manager = MultiUserTrafficManager(num_users=num_users, config=config)

    users = [User(id=0, position=np.array([0.0, 0.0]), cpu_frequency=1e9)]

    # First step should enqueue one task
    manager.step(users, dt=0.1)

    assert manager.has_pending_task(0) is True
    assert manager.get_queue_length(0) == 1
    assert manager.get_total_pending() == 1

    # Peek should not remove
    peeked = manager.peek_task(0)
    assert isinstance(peeked, Task)
    assert manager.get_queue_length(0) == 1

    # get_pending_task should remove
    task = manager.get_pending_task(0)
    assert isinstance(task, Task)
    assert manager.get_queue_length(0) == 0
    assert manager.get_total_pending() == 0

    stats = manager.get_statistics()
    assert "total_tasks" in stats
    assert "tasks_per_user" in stats
    assert stats["tasks_per_user"][0] == 1
