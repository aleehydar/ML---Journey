def calculate_breakeven_months(setup_cost_pkr: int, monthly_cost_pkr: int, estimated_monthly_revenue_pkr: int) -> int:
    """
    Calculates the breakeven point in months.
    If monthly revenue is less than or equal to monthly costs, returns -1 (never breaks even).
    """
    monthly_profit = estimated_monthly_revenue_pkr - monthly_cost_pkr
    if monthly_profit <= 0:
        return -1
    return int(setup_cost_pkr / monthly_profit) + 1

def check_budget_sufficient(budget_pkr: int, setup_cost_pkr: int, monthly_cost_pkr: int, buffer_months: int = 3) -> bool:
    """
    Checks if the user's budget is sufficient to cover setup costs plus a few months of operational runway.
    """
    required_capital = setup_cost_pkr + (monthly_cost_pkr * buffer_months)
    return budget_pkr >= required_capital
