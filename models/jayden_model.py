# models/jayden_model.py

class JaydenMethod:
    def calc_delay_score(self, delay_seconds):
        """
        Delay Score:
          0 min delay   → 1
          7.5 min delay → 5.5
          15+ min delay → 10
        """
        delay_min = delay_seconds / 60
        if delay_min <= 0:  return 1.0
        if delay_min >= 15: return 10.0
        return round(1 + (delay_min / 15) * 9, 2)

    def calc_congestion_score(self, gap_seconds):
        """
        Congestion Pressure Score:
          0 min gap  → 1
          15 min gap → 5.5
          30+ min gap → 10
        """
        gap_min = gap_seconds / 60
        if gap_min <= 0:  return 1.0
        if gap_min >= 30: return 10.0
        return round(1 + (gap_min / 30) * 9, 2)

    def jayden_method(self, delay_seconds, gap_seconds):
        """
        Final Severity Score:
        Delay(0.7) + Congestion(0.3)
        Returns score 1-10
        """
        d = self.calc_delay_score(delay_seconds)
        c = self.calc_congestion_score(gap_seconds)
        return round(d * 0.7 + c * 0.3, 2)