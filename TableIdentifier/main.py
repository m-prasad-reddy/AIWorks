# main.py
from typing import Dict, List
from database.connection import DatabaseConnection
from config.manager import DBConfigManager
from schema.manager import SchemaManager
from feedback.manager import FeedbackManager
from analysis.table_identifier import TableIdentifier
from cli.interface import DatabaseAnalyzerCLI

class DatabaseAnalyzer:
    def __init__(self):
        self.connection_manager = DatabaseConnection()
        self.config_manager = DBConfigManager()
        self.schema_dict = {}
        self.feedback_manager = None
        self.table_identifier = None
    
    def run(self):
        cli = DatabaseAnalyzerCLI(self)
        cli.run()
    
    def load_configs(self, config_path: str) -> Dict:
        return self.config_manager.load_configs(config_path)
    
    def set_current_config(self, config: Dict):
        self.connection_manager.current_config = config
    
    def connect_to_database(self):
        if self.connection_manager.connect(self.connection_manager.current_config):
            self._initialize_schema_manager()
    
    def close_connection(self):
        self.connection_manager.close()
    
    def _initialize_schema_manager(self):
        schema_manager = SchemaManager(self.connection_manager.current_config['database'])
        if schema_manager.needs_refresh(self.connection_manager.connection):
            print("\nUpdating schema cache...")
            self.schema_dict = schema_manager.build_data_dict(self.connection_manager.connection)
            print("Schema cache updated successfully!")
        else:
            print("\nLoading schema from cache...")
            self.schema_dict = schema_manager.load_from_cache()
            print("Schema loaded from cache.")
        
        self.feedback_manager = FeedbackManager(self.connection_manager.current_config['database'])
        self.table_identifier = TableIdentifier(self.schema_dict, self.feedback_manager)
    
    def process_query(self, query: str):
        try:
            identified_tables, should_display = self.table_identifier.identify_tables(query)
            
            if not should_display:
                print("\n[System] Low confidence in table identification. Please provide tables manually.")
                correct_tables = self._get_validated_tables_input()
                if correct_tables:
                    self._handle_feedback(query, correct_tables)
                    self._generate_ddl(correct_tables)
                return
            
            print("\n=== Table Identification Results ===")
            print(f"Query: {query}")
            print("\nIdentified Tables:")
            for i, table in enumerate(identified_tables, 1):
                print(f"{i}. {table}")
            
            self._handle_user_feedback(query, identified_tables)
        
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
    
    def _handle_user_feedback(self, query: str, identified_tables: List[str]):
        while True:
            feedback = input("\nIs this correct? (Y/N/back): ").strip().lower()
            
            if feedback == 'back':
                break
            elif feedback == 'y':
                valid_tables = self._validate_tables_exist(identified_tables)
                if valid_tables:
                    self._generate_ddl(valid_tables)
                    self.table_identifier.update_weights_from_feedback(query, valid_tables)
                break
            elif feedback == 'n':
                correct_tables = self._get_validated_tables_input()
                if correct_tables:
                    self._handle_feedback(query, correct_tables)
                    self._generate_ddl(correct_tables)
                break
            else:
                print("Invalid input. Please enter Y, N, or back")
    
    def _handle_feedback(self, query: str, correct_tables: List[str]):
        self.feedback_manager.store_feedback(query, correct_tables, self.schema_dict)
        self.table_identifier.update_weights_from_feedback(query, correct_tables)
    
    def _get_validated_tables_input(self) -> List[str]:
        while True:
            tables_input = input("\nEnter tables (schema.table, comma separated or 'back': ").strip()
            if tables_input.lower() == 'back':
                return []
                
            tables = [t.strip() for t in tables_input.split(',') if t.strip()]
            valid_tables, invalid_tables = self.feedback_manager.validate_tables(tables, self.schema_dict)
            
            if not invalid_tables:
                return valid_tables
                
            print(f"\n[Error] Invalid tables: {', '.join(invalid_tables)}")
            print("Available schemas and tables:")
            for schema in self.schema_dict['tables']:
                print(f"  {schema}: {', '.join(self.schema_dict['tables'][schema].keys())}")
    
    def _validate_tables_exist(self, tables: List[str]) -> List[str]:
        valid_tables = []
        invalid_tables = []
        
        schema_map = {s.lower(): s for s in self.schema_dict['tables']}
        table_maps = {
            s: {t.lower(): t for t in self.schema_dict['tables'][s]} 
            for s in self.schema_dict['tables']
        }
        
        for table in tables:
            parts = table.split('.')
            if len(parts) != 2:
                invalid_tables.append(table)
                continue
                
            schema_part, table_part = parts
            schema_lower = schema_part.lower()
            table_lower = table_part.lower()
            
            if schema_lower not in schema_map:
                invalid_tables.append(table)
                continue
                
            actual_schema = schema_map[schema_lower]
            
            if table_lower not in table_maps[actual_schema]:
                invalid_tables.append(table)
                continue
                
            actual_table = table_maps[actual_schema][table_lower]
            valid_tables.append(f"{actual_schema}.{actual_table}")
        
        if invalid_tables:
            print(f"\n[Warning] These tables don't exist in schema: {', '.join(invalid_tables)}")
            print("Please provide correct tables")
            return self._get_validated_tables_input()
            
        return valid_tables
    
    def _generate_ddl(self, tables: List[str]):
        print("\n=== Generated DDL ===")
        for table in tables:
            if '.' not in table:
                print(f"\n/* Invalid table format: {table} (should be schema.table) */")
                continue
                
            schema, table_name = table.split('.')
            print(f"\n-- Structure for [{schema}].[{table_name}]")
            
            try:
                if schema not in self.schema_dict['tables']:
                    print(f"/* Schema {schema} not found */")
                    continue
                    
                if table_name not in self.schema_dict['tables'][schema]:
                    print(f"/* Table {table_name} not found in schema {schema} */")
                    continue
                
                print(f"CREATE TABLE [{schema}].[{table_name}] (")
                
                columns = self.schema_dict['columns'][schema][table_name]
                col_defs = []
                for col_name, col_info in columns.items():
                    col_def = f"    [{col_name}] {col_info['type']}"
                    if 'is_primary_key' in col_info and col_info['is_primary_key']:
                        col_def += " PRIMARY KEY"
                    if 'nullable' in col_info and not col_info['nullable']:
                        col_def += " NOT NULL"
                    col_defs.append(col_def)
                
                print(",\n".join(col_defs))
                print(");")
                
            except Exception as e:
                print(f"/* Error generating DDL for {table}: {str(e)} */")

if __name__ == "__main__":
    analyzer = DatabaseAnalyzer()
    analyzer.run()