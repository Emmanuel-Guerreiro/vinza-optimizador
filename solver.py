import itertools
import pulp
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel

# -------------------------
# Pydantic Models
# -------------------------
class EventInstanceEntry(BaseModel):
    event: int
    day: int
    start: int
    end: int
    loc: str

# -------------------------
# Optimizer Class
# -------------------------
class Optimizer:
    def __init__(self, occurrences: Dict[str, EventInstanceEntry], travel_distances: Dict[Tuple[str, str], int], buffer_minutes: int = 60):
        """
        Initialize the Optimizer with occurrences and travel distances.
        
        Args:
            occurrences: Dictionary mapping occurrence keys to EventInstanceEntry objects
            travel_distances: Dictionary mapping (from_loc, to_loc) tuples to travel time in minutes
            buffer_minutes: Buffer time between events in minutes
        """
        self.occurrences = occurrences
        self.travel_distances = travel_distances
        self.buffer_minutes = buffer_minutes
        self.H = 60  # minutes per hour
        
        # Extract unique lists
        self.K = list(occurrences.keys())
        self.events = sorted(set(entry.event for entry in occurrences.values()))
        self.days = sorted(set(entry.day for entry in occurrences.values()))
        self.locs = sorted(set(entry.loc for entry in occurrences.values()))
        
        # Generate same day pairs
        self.same_day_pairs = [(k, m) for k, m in itertools.permutations(self.K, 2)
                              if occurrences[k].day == occurrences[m].day and k != m]
        
        # Big-M: plenty large, e.g., 24h
        self.M = 24 * self.H

    def __t(self, day: int, hh: int, mm: int = 0) -> int:
        """Convert day and time to absolute minutes from week start."""
        return day * 24 * self.H + hh * self.H + mm

    def __travel_time(self, k: str, m: str) -> int:
        """Get travel time between two occurrence locations."""
        from_loc = self.occurrences[k].loc
        to_loc = self.occurrences[m].loc
        return self.travel_distances.get((from_loc, to_loc), 0)

    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization and return the schedule.
        
        Returns:
            Dictionary containing the optimized schedule and metadata
        """
        # -------------------------
        # Model
        # -------------------------
        prob = pulp.LpProblem("MultiEventDayScheduling", pulp.LpMinimize)

        # Variables
        x = {k: pulp.LpVariable(f"x_{k}", 0, 1, cat="Binary") for k in self.K}
        u = {d: pulp.LpVariable(f"u_day{d}", 0, 1, cat="Binary") for d in self.days}
        y = {(k, m): pulp.LpVariable(f"y_{k}__{m}", 0, 1, cat="Binary") for (k, m) in self.same_day_pairs}

        # 1) Exactly one occurrence per event
        for i in self.events:
            prob += pulp.lpSum(x[k] for k in self.K if self.occurrences[k].event == i) == 1

        # 2) Activate day when any occurrence on that day is chosen
        for d in self.days:
            for k in self.K:
                if self.occurrences[k].day == d:
                    prob += x[k] <= u[d]

        # 3) Non-overlap with travel+buffer via ordering vars on same day
        for (k, m) in self.same_day_pairs:
            # link to selections
            prob += y[(k, m)] <= x[k]
            prob += y[(k, m)] <= x[m]

        # If both chosen, one of the directions must be 1 (k before m OR m before k)
        for (k, m) in [(a, b) for a, b in self.same_day_pairs if a < b and self.occurrences[a].day == self.occurrences[b].day]:
            prob += y[(k, m)] + y[(m, k)] >= x[k] + x[m] - 1
            prob += y[(k, m)] + y[(m, k)] <= 1  # at most one direction

        # Time-feasibility with big-M
        for (k, m) in self.same_day_pairs:
            ok, om = self.occurrences[k], self.occurrences[m]
            prob += om.start >= ok.end + self.__travel_time(k, m) + self.buffer_minutes - self.M * (1 - y[(k, m)])

        # Objective (lexicographic via weights)
        W1, W2, W3 = 1_000_000, 1_000, 1

        gap_cost = pulp.lpSum(
            y[(k, m)] * (self.occurrences[m].start - self.occurrences[k].end)
            for (k, m) in self.same_day_pairs
        )

        travel_cost = pulp.lpSum(
            y[(k, m)] * self.__travel_time(k, m)
            for (k, m) in self.same_day_pairs
        )

        obj = W1 * pulp.lpSum(u[d] for d in self.days) + W2 * gap_cost + W3 * travel_cost
        prob += obj

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Process solution
        chosen = [k for k in self.K if pulp.value(x[k]) > 0.5]
        schedule = {}
        
        for d in self.days:
            todays = [k for k in chosen if self.occurrences[k].day == d]
            # derive an order from y
            before_count = {k: 0 for k in todays}
            for k, m in itertools.permutations(todays, 2):
                if (k, m) in y and pulp.value(y[(k, m)]) > 0.5:
                    before_count[m] += 1
            ordered = sorted(todays, key=lambda kk: before_count[kk])
            if ordered:
                schedule[d] = ordered

        days_used = sum(int(pulp.value(u[d]) > 0.5) for d in self.days)
        
        return {
            "schedule": schedule,
            "days_used": days_used,
            "chosen_occurrences": chosen,
            "status": pulp.LpStatus[prob.status]
        }

    def print_schedule(self, result: Dict[str, Any]) -> None:
        """Print the optimized schedule in a readable format."""
        schedule = result["schedule"]
        days_used = result["days_used"]
        chosen_occurrences = result["chosen_occurrences"]
        print("Chosen occurrences:", chosen_occurrences)
        print(schedule)
        print("Days used:", days_used)
        
        for d in sorted(schedule):
            print(f"Day {d}:")
            for k in schedule[d]:
                o = self.occurrences[k]
                print(f"  - {k} | Event {o.event} | {o.start}→{o.end} | loc {o.loc}")


# -------------------------
# Sample data for testing
# -------------------------
def create_sample_data() -> Tuple[Dict[str, EventInstanceEntry], Dict[Tuple[str, str], int]]:
    """Create sample data for testing the optimizer."""
    H = 60
    BUFFER = 60
    
    def t(day, hh, mm=0):
        return day * 24 * H + hh * H + mm
    
    occurrences = {
        # Event 1
        "E1_Mon_9": EventInstanceEntry(event=1, day=0, start=t(0, 9), end=t(0, 9) + BUFFER, loc="A"),
        "E1_Mon_17": EventInstanceEntry(event=1, day=0, start=t(0, 17), end=t(0, 17) + BUFFER, loc="A"),
        "E1_Wed_16": EventInstanceEntry(event=1, day=2, start=t(2, 16), end=t(2, 16) + BUFFER, loc="A"),

        # Event 2
        "E2_Mon_10": EventInstanceEntry(event=2, day=0, start=t(0, 10), end=t(0, 10) + BUFFER, loc="B"),
        "E2_Tue_10": EventInstanceEntry(event=2, day=1, start=t(1, 10), end=t(1, 10) + BUFFER, loc="B"),

        # Event 3
        "E3_Tue_12": EventInstanceEntry(event=3, day=1, start=t(1, 12), end=t(1, 12) + BUFFER, loc="C"),
        "E3_Thu_12": EventInstanceEntry(event=3, day=3, start=t(3, 12), end=t(3, 12) + BUFFER, loc="C"),
    }

    # Travel matrix (minutes) between locations
    travel_distances = {
        ("A", "A"): 10, ("A", "B"): 25, ("A", "C"): 40,
        ("B", "A"): 25, ("B", "B"): 10, ("B", "C"): 20,
        ("C", "A"): 40, ("C", "B"): 20, ("C", "C"): 10,
    }
    
    return occurrences, travel_distances

# -------------------------
# Main function for testing
# -------------------------
def main():
    """Test the Optimizer class with sample data."""
    occurrences, travel_distances = create_sample_data()
    
    optimizer = Optimizer(occurrences, travel_distances)
    result = optimizer.optimize()
    
    print("Optimization completed!")
    print(f"Status: {result['status']}")
    print()
    
    optimizer.print_schedule(result)


if __name__ == "__main__":
    main()
