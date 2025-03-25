import json

# Example Pretraining Data
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
# Save data
with open("text_sql_pairs.json", "w") as f:
    json.dump(training_data, f)

print("Pretraining data saved.")
