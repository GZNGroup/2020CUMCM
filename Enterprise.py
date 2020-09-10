from Invoice import Invoice


class Enterprise:
    credit_rating: str
    break_contract: bool
    invoice_list: [Invoice] = []

    def __init__(self, credit_rating: str, break_contract: bool):
        self.credit_rating = credit_rating
        self.break_contract = break_contract

    def add_invoice(self, new_invoice: "Invoice"):
        self.invoice_list.append(new_invoice)

