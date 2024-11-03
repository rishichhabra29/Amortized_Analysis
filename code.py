from typing import List, Optional, Tuple
from hypothesis import given, strategies as st
import random
import logging
import matplotlib.pyplot as plt



class CostTracker:
    def __init__(self):
        self.cost = 0
        self.potential = 0
        self.cost_log = []

    def add_cost(self, units: int):
        self.cost += units
        self.cost_log.append(self.cost)

    def add_potential(self, potential_units: int):
        self.potential += potential_units

    def get_total_cost(self) -> int:
        return self.cost

    def get_total_potential(self) -> int:
        return self.potential

    def get_cost_stats(self):
        return {
            "total_cost": self.cost,
            "average_cost": sum(self.cost_log) / len(self.cost_log) if self.cost_log else 0,
            "max_cost": max(self.cost_log, default=0),
            "min_cost": min(self.cost_log, default=0),
            "variance_cost": self._calculate_variance()
        }

    def reset(self):
        self.cost = 0
        self.potential = 0
        self.cost_log.clear()

    def _calculate_variance(self) -> float:
        if len(self.cost_log) < 2:
            return 0.0
        mean = sum(self.cost_log) / len(self.cost_log)
        return sum((x - mean) ** 2 for x in self.cost_log) / (len(self.cost_log) - 1)


class SpecQueue:
    def __init__(self, cost_tracker: CostTracker, items: Optional[List[int]] = None):
        self.cost_tracker = cost_tracker
        self.items = items if items is not None else []

    def enqueue(self, item: int) -> 'SpecQueue':
        self.cost_tracker.add_cost(1)
        return SpecQueue(self.cost_tracker, self.items + [item])

    def dequeue(self) -> Tuple[Optional[int], 'SpecQueue']:
        self.cost_tracker.add_cost(1)
        if self.items:
            return self.items[0], SpecQueue(self.cost_tracker, self.items[1:])
        return None, self  # Return None if queue is empty

    def get_total_cost(self) -> str:
        return f"SpecQueue Total Cost: {self.cost_tracker.get_total_cost()}"


class BatchedQueue:
    def __init__(self, cost_tracker: CostTracker, front: Optional[List[int]] = None, back: Optional[List[int]] = None):
        self.cost_tracker = cost_tracker
        self.front = front if front is not None else []
        self.back = back if back is not None else []

    def enqueue(self, item: int) -> 'BatchedQueue':
        return BatchedQueue(self.cost_tracker, self.front, self.back + [item])

    def dequeue(self) -> Tuple[Optional[int], 'BatchedQueue']:
        if not self.front:
            self.cost_tracker.add_cost(len(self.back))  
            self.front = list(reversed(self.back))
            self.back = []

        if self.front:
            item = self.front.pop()
            self.cost_tracker.add_cost(1)  
            return item, BatchedQueue(self.cost_tracker, self.front, self.back)

        return None, self  

    def potential(self) -> int:
        return len(self.back)

    def get_total_cost(self) -> str:
        return (f"BatchedQueue Total Cost: {self.cost_tracker.get_total_cost()}, "
                f"Potential: {self.potential()}")

    def get_amortized_cost(self) -> int:
        return self.cost_tracker.get_total_cost() + self.potential()


def plot_costs(cost_log_spec: List[int], cost_log_batched: List[int], num_operations: int):
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_operations), cost_log_spec, label="SpecQueue Cost")
    plt.plot(range(num_operations), cost_log_batched, label="BatchedQueue Cost")
    plt.xlabel("Operations")
    plt.ylabel("Cumulative Cost")
    plt.title("Cost Comparison between SpecQueue and BatchedQueue")
    plt.legend()
    plt.grid()
    plt.show()  
    plt.close()  


@given(num_operations=st.integers(min_value=20, max_value=50), max_enqueue_value=st.integers(min_value=1, max_value=100))
def test_bisimulation(num_operations: int, max_enqueue_value: int):
    cost_tracker_spec = CostTracker()
    spec_queue = SpecQueue(cost_tracker_spec)

    cost_tracker_batched = CostTracker()
    batched_queue = BatchedQueue(cost_tracker_batched)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Bisimulation Test")

    logger.info("Testing Bisimulation of SpecQueue and BatchedQueue with Random Operations")

    cost_log_spec, cost_log_batched = [], []

    for i in range(num_operations):
        op = random.choice(['enqueue', 'dequeue'])

        if op == 'enqueue':
            value = random.randint(1, max_enqueue_value)
            logger.info(f"Operation {i + 1}: Enqueue {value}")
            spec_queue = spec_queue.enqueue(value)
            batched_queue = batched_queue.enqueue(value)

        elif op == 'dequeue':
            logger.info(f"Operation {i + 1}: Dequeue")
            item_spec, spec_queue = spec_queue.dequeue()
            item_batched, batched_queue = batched_queue.dequeue()
            logger.info(f"Dequeue - SpecQueue: {item_spec}, BatchedQueue: {item_batched}")

            assert item_spec == item_batched, f"Mismatch in dequeued items: {item_spec} != {item_batched}"

        cost_log_spec.append(cost_tracker_spec.get_total_cost())
        cost_log_batched.append(cost_tracker_batched.get_total_cost())

    logger.info(f"Total Operations Performed: {num_operations}")  
    logger.info("\nFinal Cost After Random Operations:")
    logger.info(spec_queue.get_total_cost())
    logger.info(batched_queue.get_total_cost())

    logger.info(f"BatchedQueue Amortized Cost: {batched_queue.get_amortized_cost()}")

    average_cost_difference = abs(cost_tracker_spec.get_total_cost() - cost_tracker_batched.get_total_cost()) / num_operations
    logger.info(f"\nAverage Cost Difference Per Operation: {average_cost_difference}")

    assert average_cost_difference < 0.5, "Average cost difference per operation is not close enough for bisimulation."

    plot_costs(cost_log_spec, cost_log_batched, num_operations)

    total_cost_difference = abs(cost_tracker_spec.get_total_cost() - cost_tracker_batched.get_total_cost())
    if total_cost_difference >= 20:
        logger.warning(f"Warning: Final cost difference is high ({total_cost_difference}) but within expected amortized bounds.")
    else:
        logger.info("Final costs are within acceptable bisimulation bounds.")

    logger.info("SpecQueue Cost Stats: %s", cost_tracker_spec.get_cost_stats())
    logger.info("BatchedQueue Cost Stats: %s", cost_tracker_batched.get_cost_stats())


if __name__ == "__main__":
    try:
        test_bisimulation()
    except KeyboardInterrupt:
        print("Process interrupted by the user. Exiting gracefully...")
    finally:
        plt.close('all')  
