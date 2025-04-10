import os
import json
import pyodbc
import numpy as np
import spacy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load English language model for NLP
nlp = spacy.load("en_core_web_sm")

class DatabaseAnalyzer:
    def __init__(self):
        self.current_config = None
        self.connection = None
        self.schema_dict = {}
        self.feedback_learner = None
        self.table_identifier = None
        
    def run(self):
        print("=== Database Schema Analyzer ===")
        print("1. Load Database Configuration")
        print("2. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-2): ").strip()
            
            if choice == "1":
                self.handle_database_config()
                if self.connection:
                    self.main_analysis_loop()
            elif choice == "2":
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def handle_database_config(self):
        config_path = input("Enter config file path [default: database_configurations.json]: ").strip()
        if not config_path:
            config_path = os.path.join(os.getcwd(), "database_configurations.json")
        
        try:
            config_manager = DBConfigManager()
            configs = config_manager.load_configs(config_path)
            
            print("\nAvailable Database Configurations:")
            for i, key in enumerate(configs.keys(), 1):
                print(f"{i}. {key}")
            print(f"{len(configs)+1}. Back to main menu")
            
            while True:
                choice = input("\nSelect configuration (number) or 'back': ").strip()
                
                if choice.lower() == 'back':
                    return
                
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(configs):
                        config_key = list(configs.keys())[choice_idx]
                        self.current_config = configs[config_key]
                        self.connect_to_database()
                        return
                    elif choice_idx == len(configs):
                        return
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        
        except Exception as e:
            print(f"Error loading configurations: {str(e)}")
    
    def connect_to_database(self):
        try:
            conn_str = (
                f"DRIVER={{{self.current_config['driver']}}};"
                f"SERVER={self.current_config['server']};"
                f"DATABASE={self.current_config['database']};"
                f"UID={self.current_config['username']};"
                f"PWD={self.current_config['password']}"
            )
            self.connection = pyodbc.connect(conn_str)
            print(f"\nSuccessfully connected to {self.current_config['database']}!")
            
            # Initialize schema and feedback systems
            self.initialize_schema_manager()
            self.feedback_learner = FeedbackLearner(self.current_config['database'])
            self.table_identifier = TableIdentifier(self.schema_dict, self.feedback_learner)
            
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            self.connection = None
    
    def initialize_schema_manager(self):
        schema_manager = SchemaManager(self.current_config['database'])
        if schema_manager.needs_refresh(self.connection):
            print("Updating schema cache...")
            self.schema_dict = schema_manager.build_data_dict(self.connection)
            print("Schema cache updated successfully!")
        else:
            print("Loading schema from cache...")
            self.schema_dict = schema_manager.load_from_cache()
            print("Schema loaded from cache.")
    
    def main_analysis_loop(self):
        while True:
            print("\nOptions:")
            print("1. Enter a query for table identification")
            print("2. Switch database configuration")
            print("3. Back to main menu")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                self.process_user_query()
            elif choice == "2":
                self.handle_database_config()
                if not self.connection:
                    return  # Back to main menu
            elif choice == "3":
                if self.connection:
                    self.connection.close()
                    self.connection = None
                return
            else:
                print("Invalid choice. Please try again.")
    
    def process_user_query(self):
        query = input("\nEnter your database query in natural language: ").strip()
        if not query:
            print("No query entered.")
            return
        
        print("\nAnalyzing query...")
        
        try:
            # Step 1: Identify tables
            identified_tables = self.table_identifier.identify_tables(query)
            
            # Step 2: Present analysis
            print("\n=== Table Identification Results ===")
            print(f"Query: {query}")
            print("\nIdentified Tables:")
            for i, table in enumerate(identified_tables, 1):
                print(f"{i}. {table}")
            
            # Step 3: Get user feedback
            feedback = input("\nIs this correct? (Y/N, or 'skip'): ").strip().lower()
            
            if feedback == 'y':
                print("\nProceeding with identified tables...")
                self.generate_ddl(identified_tables)
            elif feedback == 'n':
                correct_tables = input("Enter correct tables (comma separated): ").strip().split(',')
                correct_tables = [t.strip() for t in correct_tables if t.strip()]
                if correct_tables:
                    self.feedback_learner.store_feedback(query, correct_tables)
                    print("\nThank you for your feedback! The system will learn from this.")
                    self.generate_ddl(correct_tables)
                else:
                    print("\nNo valid tables provided. Feedback not stored.")
            else:
                print("\nSkipping feedback for this query.")
        
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
    
    def generate_ddl(self, tables: List[str]):
        print("\n=== Generated DDL ===")
        for table in tables:
            schema, table_name = table.split('.') if '.' in table else (None, table)
            print(f"\n-- Structure for {table}")
            
            try:
                # Get table structure from schema dictionary
                table_info = None
                if schema:
                    table_info = self.schema_dict['tables'].get(schema, {}).get(table_name)
                else:
                    for schema_name in self.schema_dict['tables']:
                        if table_name in self.schema_dict['tables'][schema_name]:
                            table_info = self.schema_dict['tables'][schema_name][table_name]
                            schema = schema_name
                            break
                
                if not table_info:
                    print(f"/* Table {table} not found in schema */")
                    continue
                
                # Generate CREATE TABLE statement
                print(f"CREATE TABLE {schema}.{table_name} (")
                
                columns = self.schema_dict['columns'][schema][table_name]
                col_defs = []
                for col_name, col_info in columns.items():
                    col_def = f"    {col_name} {col_info['type']}"
                    if 'is_primary_key' in col_info and col_info['is_primary_key']:
                        col_def += " PRIMARY KEY"
                    if 'nullable' in col_info and not col_info['nullable']:
                        col_def += " NOT NULL"
                    col_defs.append(col_def)
                
                print(",\n".join(col_defs))
                print(");")
                
            except Exception as e:
                print(f"/* Error generating DDL for {table}: {str(e)} */")

class DBConfigManager:
    def load_configs(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path) as f:
            configs = json.load(f)
        
        if not isinstance(configs, dict):
            raise ValueError("Config file should contain a dictionary of configurations")
        
        required_keys = {'server', 'database', 'username', 'password', 'driver'}
        for key, config in configs.items():
            if not isinstance(config, dict):
                raise ValueError(f"Configuration for {key} must be a dictionary")
            if not required_keys.issubset(config.keys()):
                missing = required_keys - set(config.keys())
                raise ValueError(f"Missing keys in {key} config: {', '.join(missing)}")
        
        return configs

class SchemaManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.cache_dir = os.path.join("schema_cache", db_name)
        self.cache_file = os.path.join(self.cache_dir, "schema.json")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def needs_refresh(self, conn) -> bool:
        if not os.path.exists(self.cache_file):
            return True
        
        cached_time = os.path.getmtime(self.cache_file)
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT MAX(STATS_DATE(object_id, stats_id)) as last_updated
                    FROM sys.stats
                    WHERE object_id IN (
                        SELECT object_id 
                        FROM sys.tables 
                        WHERE is_ms_shipped = 0
                    )
                """)
                result = cursor.fetchone()
                latest_change = result[0].timestamp() if result[0] else 0
                
                return latest_change > cached_time
        
        except Exception:
            return True
    
    def build_data_dict(self, conn) -> Dict:
        schema_dict = {
            "schemas": {},
            "tables": defaultdict(dict),
            "columns": defaultdict(lambda: defaultdict(dict)),
            "relationships": []
        }
        
        try:
            with conn.cursor() as cursor:
                # Get schema information
                cursor.execute("""
                    SELECT name, schema_id
                    FROM sys.schemas
                    WHERE principal_id = 1
                """)
                for row in cursor.fetchall():
                    schema_dict["schemas"][row.name] = {
                        "id": row.schema_id,
                        "tables": []
                    }
                
                # Get table information
                cursor.execute("""
                    SELECT t.name AS table_name, s.name AS schema_name, t.object_id
                    FROM sys.tables t
                    JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE t.is_ms_shipped = 0
                """)
                for row in cursor.fetchall():
                    schema_dict["tables"][row.schema_name][row.table_name] = {
                        "id": row.object_id,
                        "columns": []
                    }
                    schema_dict["schemas"][row.schema_name]["tables"].append(row.table_name)
                
                # Get column information
                cursor.execute("""
                    SELECT 
                        s.name AS schema_name,
                        t.name AS table_name,
                        c.name AS column_name,
                        c.column_id,
                        ty.name AS type_name,
                        c.max_length,
                        c.precision,
                        c.scale,
                        c.is_nullable,
                        c.is_identity,
                        ep.value AS description
                    FROM sys.columns c
                    JOIN sys.tables t ON c.object_id = t.object_id
                    JOIN sys.schemas s ON t.schema_id = s.schema_id
                    JOIN sys.types ty ON c.user_type_id = ty.user_type_id
                    LEFT JOIN sys.extended_properties ep ON 
                        ep.major_id = c.object_id AND 
                        ep.minor_id = c.column_id AND
                        ep.name = 'MS_Description'
                    WHERE t.is_ms_shipped = 0
                """)
                for row in cursor.fetchall():
                    col_info = {
                        "id": row.column_id,
                        "type": row.type_name,
                        "max_length": row.max_length,
                        "precision": row.precision,
                        "scale": row.scale,
                        "nullable": bool(row.is_nullable),
                        "identity": bool(row.is_identity),
                        "description": row.description
                    }
                    schema_dict["columns"][row.schema_name][row.table_name][row.column_name] = col_info
                    schema_dict["tables"][row.schema_name][row.table_name]["columns"].append(row.column_name)
                
                # Get primary keys
                cursor.execute("""
                    SELECT 
                        s.name AS schema_name,
                        t.name AS table_name,
                        c.name AS column_name
                    FROM sys.key_constraints kc
                    JOIN sys.tables t ON kc.parent_object_id = t.object_id
                    JOIN sys.schemas s ON t.schema_id = s.schema_id
                    JOIN sys.index_columns ic ON 
                        ic.object_id = kc.parent_object_id AND
                        ic.index_id = kc.unique_index_id
                    JOIN sys.columns c ON 
                        ic.object_id = c.object_id AND
                        ic.column_id = c.column_id
                    WHERE kc.type = 'PK'
                """)
                for row in cursor.fetchall():
                    if row.column_name in schema_dict["columns"][row.schema_name][row.table_name]:
                        schema_dict["columns"][row.schema_name][row.table_name][row.column_name]["is_primary_key"] = True
                
                # Get foreign key relationships
                cursor.execute("""
                    SELECT 
                        fs.name AS from_schema,
                        ft.name AS from_table,
                        fc.name AS from_column,
                        ts.name AS to_schema,
                        tt.name AS to_table,
                        tc.name AS to_column
                    FROM sys.foreign_key_columns fkc
                    JOIN sys.tables ft ON fkc.parent_object_id = ft.object_id
                    JOIN sys.schemas fs ON ft.schema_id = fs.schema_id
                    JOIN sys.columns fc ON 
                        fkc.parent_object_id = fc.object_id AND
                        fkc.parent_column_id = fc.column_id
                    JOIN sys.tables tt ON fkc.referenced_object_id = tt.object_id
                    JOIN sys.schemas ts ON tt.schema_id = ts.schema_id
                    JOIN sys.columns tc ON 
                        fkc.referenced_object_id = tc.object_id AND
                        fkc.referenced_column_id = tc.column_id
                """)
                for row in cursor.fetchall():
                    relationship = {
                        "from": f"{row.from_schema}.{row.from_table}.{row.from_column}",
                        "to": f"{row.to_schema}.{row.to_table}.{row.to_column}",
                        "cross_schema": row.from_schema != row.to_schema
                    }
                    schema_dict["relationships"].append(relationship)
                
                # Save to cache
                with open(self.cache_file, 'w') as f:
                    json.dump(schema_dict, f, indent=2)
                
                return schema_dict
        
        except Exception as e:
            print(f"Error building schema dictionary: {str(e)}")
            raise
    
    def load_from_cache(self) -> Dict:
        with open(self.cache_file) as f:
            return json.load(f)

class FeedbackLearner:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.feedback_dir = os.path.join("feedback", db_name)
        os.makedirs(self.feedback_dir, exist_ok=True)
    
    def store_feedback(self, query: str, correct_tables: List[str]):
        try:
            embedding = self.model.encode(query)
            feedback_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            np.save(os.path.join(self.feedback_dir, f"{feedback_id}_emb.npy"), embedding)
            with open(os.path.join(self.feedback_dir, f"{feedback_id}_meta.json"), 'w') as f:
                json.dump({
                    'query': query,
                    'tables': correct_tables,
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            print(f"\n[System] Feedback stored (ID: {feedback_id})")
        
        except Exception as e:
            print(f"\n[System] Error storing feedback: {str(e)}")
    
    def get_similar_feedback(self, query: str, threshold: float = 0.85) -> Optional[List[Dict]]:
        try:
            query_emb = self.model.encode(query).reshape(1, -1)
            feedback_items = []
            
            for fname in os.listdir(self.feedback_dir):
                if fname.endswith("_emb.npy"):
                    emb_path = os.path.join(self.feedback_dir, fname)
                    meta_path = emb_path.replace("_emb.npy", "_meta.json")
                    
                    if os.path.exists(meta_path):
                        stored_emb = np.load(emb_path)
                        similarity = cosine_similarity(
                            query_emb, 
                            stored_emb.reshape(1, -1)
                        )[0][0]
                        
                        if similarity >= threshold:
                            with open(meta_path) as f:
                                meta = json.load(f)
                            feedback_items.append({
                                "similarity": similarity,
                                "query": meta["query"],
                                "tables": meta["tables"],
                                "timestamp": meta["timestamp"]
                            })
            
            feedback_items.sort(key=lambda x: x["similarity"], reverse=True)
            return feedback_items if feedback_items else None
        
        except Exception as e:
            print(f"\n[System] Error retrieving feedback: {str(e)}")
            return None

class TableIdentifier:
    def __init__(self, schema_dict: Dict, feedback_learner: FeedbackLearner):
        self.schema_dict = schema_dict
        self.feedback_learner = feedback_learner
        self.cross_schema_index = self.build_cross_schema_index()
    
    def build_cross_schema_index(self) -> Dict:
        column_map = defaultdict(list)
        
        for schema_name, tables in self.schema_dict["columns"].items():
            for table_name, columns in tables.items():
                for column_name in columns:
                    full_name = f"{schema_name}.{table_name}.{column_name}"
                    column_map[column_name.lower()].append(full_name)
        
        return {k: v for k, v in column_map.items() if len(v) > 1}
    
    def identify_tables(self, query: str) -> List[str]:
        if (feedback := self.feedback_learner.get_similar_feedback(query)):
            print("\n[System] Found similar past queries in feedback database:")
            for i, item in enumerate(feedback[:3], 1):
                print(f"{i}. Similarity: {item['similarity']:.2f}")
                print(f"   Query: {item['query']}")
                print(f"   Tables: {', '.join(item['tables'])}")
            
            return feedback[0]["tables"]
        
        print("\n[System] No relevant feedback found. Analyzing query with NLP...")
        entities = self.extract_entities(query)
        tables = self.find_matching_tables(entities)
        tables = self.resolve_cross_schema(tables)
        
        return tables
    
    def extract_entities(self, query: str) -> List[Tuple[str, str]]:
        doc = nlp(query)
        entities = []
        
        for chunk in doc.noun_chunks:
            entities.append((chunk.text.lower(), "table_candidate"))
        
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                entities.append((ent.text.lower(), "temporal"))
            elif ent.label_ == "MONEY":
                entities.append((ent.text.lower(), "monetary"))
        
        for token in doc:
            if token.pos_ == "VERB":
                entities.append((token.lemma_.lower(), "action"))
        
        return entities
    
    def find_matching_tables(self, entities: List[Tuple[str, str]]) -> List[str]:
        matched_tables = set()
        
        for entity_text, entity_type in entities:
            for schema_name, tables in self.schema_dict["tables"].items():
                for table_name in tables:
                    if entity_text in table_name.lower():
                        matched_tables.add(f"{schema_name}.{table_name}")
            
            for schema_name, tables in self.schema_dict["columns"].items():
                for table_name, columns in tables.items():
                    for column_name in columns:
                        if entity_text in column_name.lower():
                            matched_tables.add(f"{schema_name}.{table_name}")
        
        return list(matched_tables)
    
    def resolve_cross_schema(self, tables: List[str]) -> List[str]:
        expanded_tables = set(tables)
        
        for rel in self.schema_dict["relationships"]:
            from_parts = rel["from"].split('.')
            to_parts = rel["to"].split('.')
            
            from_table = f"{from_parts[0]}.{from_parts[1]}"
            to_table = f"{to_parts[0]}.{to_parts[1]}"
            
            if from_table in expanded_tables and to_table not in expanded_tables:
                expanded_tables.add(to_table)
            elif to_table in expanded_tables and from_table not in expanded_tables:
                expanded_tables.add(from_table)
        
        return list(expanded_tables)

if __name__ == "__main__":
    analyzer = DatabaseAnalyzer()
    analyzer.run()