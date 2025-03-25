from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments
import torch
import json

training_data = [
    {"text": "show all sales orders", 
     "sql": "SELECT * FROM adventureworks.SalesLT.SalesOrderHeader;",
     "schema": "Tables: SalesLT.SalesOrderHeader | Columns: SalesOrderID, RevisionNumber, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag, SalesOrderNumber, PurchaseOrderNumber, AccountNumber, CustomerID, ShipToAddressID, BillToAddressID, ShipMethod, CreditCardApprovalCode, SubTotal, TaxAmt, Freight, TotalDue, Comment, rowguid, ModifiedDate"
     },
    {"text": "Find orders placed in 2021", 
     "sql": "SELECT * FROM adventureworks.SalesLT.SalesOrderHeader WHERE OrderDate >= '2021-01-01';",
     "schema": "Tables: SalesLT.SalesOrderHeader | Columns: SalesOrderID, RevisionNumber, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag, SalesOrderNumber, PurchaseOrderNumber, AccountNumber, CustomerID, ShipToAddressID, BillToAddressID, ShipMethod, CreditCardApprovalCode, SubTotal, TaxAmt, Freight, TotalDue, Comment, rowguid, ModifiedDate"
     },
# ] 
# training_data = [
    {
        "text": "show all addresses",
        "sql": "SELECT  AddressID, AddressLine1, AddressLine2, City, StateProvince, CountryRegion, PostalCode, rowguid, ModifiedDate FROM adventureworks.SalesLT.Address;",
        "schema": "Tables: SalesLT.Address | Columns: AddressID, AddressLine1, AddressLine2, City, StateProvince, CountryRegion, PostalCode, rowguid, ModifiedDate"
    },
	{
        "text": "show all Customer addresses",
        "sql": "SELECT  CustomerID, AddressID, AddressType, rowguid, ModifiedDate FROM adventureworks.SalesLT.CustomerAddress;",
        "schema": "Tables: SalesLT.CustomerAddress | Columns: CustomerID, AddressID, AddressType, rowguid, ModifiedDate"
    },
    {
        "text": "Find orders placed after 2007",
        "sql": "SELECT * FROM SalesLT.SalesOrderHeader WHERE OrderDate >= '2007-01-01';",
        "schema": "Tables: SalesLT.SalesOrderHeader | Columns: SalesOrderID, RevisionNumber, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag, SalesOrderNumber, PurchaseOrderNumber, AccountNumber, CustomerID, ShipToAddressID, BillToAddressID, ShipMethod, CreditCardApprovalCode, SubTotal, TaxAmt, Freight, TotalDue, Comment, rowguid, ModifiedDate"
    }
]

# Load Pretraining Data
with open("text_sql_pairs.json", "w") as f:
    json.dump(training_data, f)

#Fixing the error with updated training data
input_texts = [f"Schema: {d['schema']} | Question: {d['text']}" for d in training_data]
target_texts = [d['sql'] for d in training_data]


model_name = "salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


training_args = TrainingArguments(
    output_dir="./sql_model2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,  # Train longer for better results
    logging_dir="./logs"
)

def generate_sql(query_text, schema):
    input_text = f"Schema: {schema} | Question: {query_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
schema = "Tables: SalesLT.SalesOrderHeader | Columns: SalesOrderID, RevisionNumber, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag, SalesOrderNumber, PurchaseOrderNumber, AccountNumber, CustomerID, ShipToAddressID, BillToAddressID, ShipMethod, CreditCardApprovalCode, SubTotal, TaxAmt, Freight, TotalDue, Comment, rowguid, ModifiedDate"
query = "Show sales orders after 2007."
print(generate_sql(query, schema))
