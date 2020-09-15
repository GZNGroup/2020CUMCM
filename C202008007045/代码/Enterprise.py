from Invoice import Invoice


class Enterprise:
    number: str
    name: str
    credit_rating: str
    break_contract: bool
    invoice_list: ["Invoice"]

    def __init__(
        self, number: str, name: str, credit_rating: str, break_contract: bool
    ):
        self.number = number
        self.name = name
        self.credit_rating = credit_rating
        self.break_contract = break_contract
        self.invoice_list = []

    def add_invoice(self, new_invoice: "Invoice"):
        self.invoice_list.append(new_invoice)

