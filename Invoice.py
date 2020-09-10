from datetime import date
from Enterprise import Enterprise


class Invoice:
    number: int
    date: "date"
    self_enterprise: "Enterprise"
    partner: str
    amount: float
    tax: float
    sum_money: float
    state_avaliable: bool

    def __init__(
        self,
        number: int,
        date: "date",
        self_enterprise: "Enterprise",
        partner: str,
        amount: float,
        tax: float,
        sum_money: float,
        state_avaliable: bool,
    ):
        self.number = number
        self.date = date
        self.self_enterprise = self_enterprise
        self.partner = partner
        self.amount = amount
        self.tax = tax
        self.sum_money = sum_money
        self.state_avaliable = state_avaliable
