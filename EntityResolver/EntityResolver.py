import os
import json
import re
import numpy as np
import spacy
import pyodbc
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, DefaultDict
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
        while True:
            print("\nMain Menu:")
            print("1. Load Database Configuration")
            print("2. Exit")
            
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
        config_path = input("\nEnter config file path [default: database_configurations.json]: ").strip()
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
            print("\nUpdating schema cache...")
            self.schema_dict = schema_manager.build_data_dict(self.connection)
            print("Schema cache updated successfully!")
        else:
            print("\nLoading schema from cache...")
            self.schema_dict = schema_manager.load_from_cache()
            print("Schema loaded from cache.")
    
    def main_analysis_loop(self):
        while True:
            print("\nOptions:")
            print("1. Enter query mode")
            print("2. Switch database configuration")
            print("3. Back to main menu")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                self.query_mode()
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
    
    def query_mode(self):
        while True:
            query = input("\nEnter your database query (or 'back' to return to options): ").strip()
            if query.lower() == 'back':
                return
            
            if not query:
                print("No query entered.")
                continue
            
            print("\nAnalyzing query...")
            
            try:
                identified_tables, should_display = self.table_identifier.identify_tables(query)
                
                if not should_display:
                    print("\n[System] Low confidence in table identification. Please provide tables manually.")
                    correct_tables = self._get_validated_tables_input()
                    if correct_tables:
                        self.feedback_learner.store_feedback(query, correct_tables, self.schema_dict)
                        self.table_identifier.update_weights_from_feedback(query, correct_tables)
                        self.generate_ddl(correct_tables)
                    continue
                
                print("\n=== Table Identification Results ===")
                print(f"Query: {query}")
                print("\nIdentified Tables:")
                for i, table in enumerate(identified_tables, 1):
                    print(f"{i}. {table}")
                
                while True:
                    feedback = input("\nIs this correct? (Y/N/back): ").strip().lower()
                    
                    if feedback == 'back':
                        break
                    elif feedback == 'y':
                        valid_tables = self._validate_tables_exist(identified_tables)
                        if valid_tables:
                            self.generate_ddl(valid_tables)
                            self.table_identifier.update_weights_from_feedback(query, valid_tables)
                        break
                    elif feedback == 'n':
                        correct_tables = self._get_validated_tables_input()
                        if correct_tables:
                            self.feedback_learner.store_feedback(query, correct_tables, self.schema_dict)
                            self.table_identifier.update_weights_from_feedback(query, correct_tables)
                            self.generate_ddl(correct_tables)
                        break
                    else:
                        print("Invalid input. Please enter Y, N, or back")
            
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
    
    def _get_validated_tables_input(self) -> List[str]:
        """Get and validate table input from user"""
        while True:
            tables_input = input("\nEnter tables (schema.table, comma separated or 'back'): ").strip()
            if tables_input.lower() == 'back':
                return []
                
            tables = [t.strip() for t in tables_input.split(',') if t.strip()]
            
            valid_tables, invalid_tables = self.feedback_learner.validate_tables(
                tables, self.schema_dict)
            
            if not invalid_tables:
                return valid_tables
                
            print(f"\n[Error] Invalid tables: {', '.join(invalid_tables)}")
            print("Available schemas and tables:")
            for schema in self.schema_dict['tables']:
                print(f"  {schema}: {', '.join(self.schema_dict['tables'][schema].keys())}")
    
    def _validate_tables_exist(self, tables: List[str]) -> List[str]:
        """Ensure tables exist in schema (case-insensitive)"""
        valid_tables = []
        invalid_tables = []
        
        # Build lowercase mapping of all schemas and tables
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
            
            # Find matching schema (case-insensitive)
            if schema_lower not in schema_map:
                invalid_tables.append(table)
                continue
                
            actual_schema = schema_map[schema_lower]
            
            # Find matching table (case-insensitive)
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
    
    def generate_ddl(self, tables: List[str]):
        print("\n=== Generated DDL ===")
        for table in tables:
            if '.' not in table:
                print(f"\n/* Invalid table format: {table} (should be schema.table) */")
                continue
                
            schema, table_name = table.split('.')
            print(f"\n-- Structure for [{schema}].[{table_name}]")
            
            try:
                # Verify table exists
                if schema not in self.schema_dict['tables']:
                    print(f"/* Schema {schema} not found */")
                    continue
                    
                if table_name not in self.schema_dict['tables'][schema]:
                    print(f"/* Table {table_name} not found in schema {schema} */")
                    continue
                
                # Generate CREATE TABLE statement
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
            "database": self.db_name,
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
                
                # Generate initial weights if they don't exist
                weights_file = os.path.join(self.cache_dir, "weights.json")
                if not os.path.exists(weights_file):
                    TableIdentifier(schema_dict, None)  # This will generate weights
                
                return schema_dict
        
        except Exception as e:
            print(f"Error building schema dictionary: {str(e)}")
            raise
    
    def load_from_cache(self) -> Dict:
        with open(self.cache_file) as f:
            schema_dict = json.load(f)
            # Ensure database name is set
            schema_dict['database'] = self.db_name
            return schema_dict

class FeedbackLearner:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.feedback_dir = os.path.join("feedback", db_name)
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.feedback_cache = {}  # In-memory cache for faster lookup
        self.pattern_cache = {}   # Cache for query patterns
        self._load_feedback_cache()
    
    def _load_feedback_cache(self):
        """Load all feedback into memory for faster access"""
        self.feedback_cache.clear()
        self.pattern_cache.clear()
        
        for fname in os.listdir(self.feedback_dir):
            if fname.endswith("_meta.json"):
                with open(os.path.join(self.feedback_dir, fname)) as f:
                    meta = json.load(f)
                    # Normalize table names to lowercase for consistency
                    normalized_tables = [t.lower() for t in meta['tables']]
                    
                    # Store exact match
                    self.feedback_cache[meta['query'].lower()] = {
                        'tables': normalized_tables,
                        'timestamp': meta['timestamp']
                    }
                    
                    # Store pattern match
                    pattern = self._extract_query_pattern(meta['query'])
                    if pattern not in self.pattern_cache:
                        self.pattern_cache[pattern] = {
                            'tables': normalized_tables,
                            'timestamp': meta['timestamp'],
                            'count': 1
                        }
                    else:
                        self.pattern_cache[pattern]['count'] += 1
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract a generalized pattern from the query with condition handling"""
        doc = nlp(query.lower())
        pattern = []
        skip_next = False
        
        for i, token in enumerate(doc):
            if skip_next:
                skip_next = False
                continue
                
            if token.text in ('=', '!=') and i > 0:
                # Found a condition, mark the pattern
                pattern.append(f"{doc[i-1].lemma_}=[CONDITION]")
                skip_next = True  # Skip the value
            elif token.like_num:
                pattern.append('[YEAR]')
            elif token.is_quote:
                pattern.append('[QUOTED]')
            else:
                pattern.append(token.lemma_)
        
        return ' '.join(pattern)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query by handling conditions, numbers, and special chars"""
        # Replace conditions with placeholders
        query = re.sub(r'(\w+)\s*[=!]=\s*[\'"]?([\w\s]+)[\'"]?', 
                      r'\1=[CONDITION]', query, flags=re.IGNORECASE)
        
        # Remove standalone numbers
        query = ' '.join([word for word in query.split() if not word.isdigit()])
        
        # Clean special chars but keep spaces
        return re.sub(r'[^a-z0-9\s]', '', query.lower())
    
    def validate_tables(self, tables: List[str], schema_dict: Dict) -> Tuple[List[str], List[str]]:
        """Validate table names against schema dictionary (case-insensitive)"""
        valid_tables = []
        invalid_tables = []
        
        # Build lowercase mapping of all schemas and tables
        schema_map = {s.lower(): s for s in schema_dict['tables']}
        table_maps = {
            s: {t.lower(): t for t in schema_dict['tables'][s]} 
            for s in schema_dict['tables']
        }
        
        for table in tables:
            parts = table.split('.')
            if len(parts) != 2:
                invalid_tables.append(table)
                continue
                
            schema_part, table_part = parts
            schema_lower = schema_part.lower()
            table_lower = table_part.lower()
            
            # Find matching schema (case-insensitive)
            if schema_lower not in schema_map:
                invalid_tables.append(table)
                continue
                
            actual_schema = schema_map[schema_lower]
            
            # Find matching table (case-insensitive)
            if table_lower not in table_maps[actual_schema]:
                invalid_tables.append(table)
                continue
                
            actual_table = table_maps[actual_schema][table_lower]
            valid_tables.append(f"{actual_schema}.{actual_table}")
                
        return valid_tables, invalid_tables
    
    def store_feedback(self, query: str, correct_tables: List[str], schema_dict: Dict) -> bool:
        """Store feedback with validation and update existing if needed"""
        valid_tables, invalid_tables = self.validate_tables(correct_tables, schema_dict)
        
        if invalid_tables:
            print(f"\n[Error] These tables don't exist: {', '.join(invalid_tables)}")
            return False
            
        # Normalize table names to lowercase for consistency
        normalized_tables = [t.lower() for t in valid_tables]
            
        # Check for existing feedback to update
        existing = self._find_exact_match(query)
        if existing:
            self._update_feedback(existing, normalized_tables)
            print("\n[System] Updated existing feedback record")
        else:
            self._create_new_feedback(query, normalized_tables)
            print("\n[System] New feedback stored")
        
        # Update the in-memory cache
        self._load_feedback_cache()
        return True
    
    def _find_exact_match(self, query: str) -> Optional[str]:
        """Find exact query match in feedback (case-insensitive)"""
        query_lower = query.lower()
        for fname in os.listdir(self.feedback_dir):
            if fname.endswith("_meta.json"):
                with open(os.path.join(self.feedback_dir, fname)) as f:
                    meta = json.load(f)
                    if meta['query'].lower() == query_lower:
                        return fname.replace("_meta.json", "")
        return None
    
    def _update_feedback(self, feedback_id: str, tables: List[str]):
        """Update existing feedback"""
        meta_path = os.path.join(self.feedback_dir, f"{feedback_id}_meta.json")
        with open(meta_path, 'r+') as f:
            meta = json.load(f)
            meta['tables'] = tables
            meta['timestamp'] = datetime.now().isoformat()
            f.seek(0)
            json.dump(meta, f)
            f.truncate()
    
    def _create_new_feedback(self, query: str, tables: List[str]):
        """Create new feedback entry"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        embedding = self.model.encode(self._normalize_query(query))
        
        np.save(os.path.join(self.feedback_dir, f"{timestamp}_emb.npy"), embedding)
        with open(os.path.join(self.feedback_dir, f"{timestamp}_meta.json"), 'w') as f:
            json.dump({
                'query': query,
                'tables': tables,
                'timestamp': datetime.now().isoformat()
            }, f)
    
    def get_similar_feedback(self, query: str, threshold: float = 0.85) -> Optional[List[Dict]]:
        """Retrieve similar feedback using exact match first, then pattern match, then semantic similarity"""
        try:
            # First try exact match (case-insensitive)
            query_lower = query.lower()
            if query_lower in self.feedback_cache:
                return [{
                    'similarity': 1.0,
                    'query': query,
                    'tables': self.feedback_cache[query_lower]['tables'],
                    'timestamp': self.feedback_cache[query_lower]['timestamp'],
                    'type': 'exact'
                }]

            # Then try pattern match
            pattern = self._extract_query_pattern(query)
            if pattern in self.pattern_cache:
                return [{
                    'similarity': 1.0,
                    'query': query,
                    'tables': self.pattern_cache[pattern]['tables'],
                    'timestamp': self.pattern_cache[pattern]['timestamp'],
                    'type': 'pattern',
                    'pattern': pattern,
                    'count': self.pattern_cache[pattern]['count']
                }]

            # Fall back to semantic similarity if no exact or pattern match
            query_emb = self.model.encode(self._normalize_query(query)).reshape(1, -1)
            feedback_items = []
            
            for fname in os.listdir(self.feedback_dir):
                if fname.endswith("_emb.npy"):
                    emb_path = os.path.join(self.feedback_dir, fname)
                    meta_path = emb_path.replace("_emb.npy", "_meta.json")
                    
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                        stored_emb = np.load(emb_path)
                        similarity = cosine_similarity(
                            query_emb, 
                            stored_emb.reshape(1, -1)
                        )[0][0]
                        
                        if similarity >= threshold:
                            feedback_items.append({
                                "similarity": similarity,
                                "query": meta["query"],
                                "tables": meta["tables"],
                                "timestamp": meta["timestamp"],
                                "type": "semantic"
                            })
            
            # Sort by similarity (highest first)
            feedback_items.sort(key=lambda x: x["similarity"], reverse=True)
            return feedback_items if feedback_items else None
        
        except Exception as e:
            print(f"\n[System] Error retrieving feedback: {str(e)}")
            return None

class TableIdentifier:
    # Confidence thresholds
    MIN_WEIGHT_TO_DISPLAY = 0.8  # Show results if above this
    MIN_WEIGHT_TO_ACCEPT = 1.0   # Auto-accept if above this
    
    def __init__(self, schema_dict: Dict, feedback_learner: FeedbackLearner):
        self.schema_dict = schema_dict
        self.feedback_learner = feedback_learner
        self.cross_schema_index = self.build_cross_schema_index()
        self.weights_file = os.path.join("schema_cache", schema_dict.get('database', 'default'), "weights.json")
        
        # Initialize weights - load or generate fresh
        self.entity_weights = self._initialize_weights()
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize weights from cache or generate new weights from schema"""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file) as f:
                    return json.load(f)
            except Exception:
                pass  # Fall through to generate fresh weights
        
        # Generate fresh weights from schema
        weights = self._generate_base_weights()
        
        # Save for future use
        os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
        with open(self.weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
            
        return weights
    
    def _generate_base_weights(self) -> Dict[str, float]:
        """Generate base weights by analyzing schema"""
        weights = defaultdict(float)
        
        # 1. Analyze table names
        for schema, tables in self.schema_dict['tables'].items():
            for table_name in tables.keys():
                # Split table names into components (handle both camelCase and snake_case)
                components = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[a-z]+', table_name.lower())
                
                # Weight components based on their frequency
                for comp in components:
                    if len(comp) > 2:  # Ignore very short components
                        weights[comp] += 1.0
                        
                        # Special boost for common suffixes
                        if comp.endswith(('s', 'es')):  # Plural forms
                            weights[comp[:-1]] += 0.5
                        
                        # Common table patterns
                        if comp in {'tbl', 'table', 'rel', 'xref', 'map', 'link'}:
                            weights[comp] += 0.3
        
        # 2. Analyze column names
        for schema, tables in self.schema_dict['columns'].items():
            for table_name, columns in tables.items():
                for column_name in columns.keys():
                    col_components = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[a-z]+', column_name.lower())
                    
                    for comp in col_components:
                        if len(comp) > 2:
                            weights[comp] += 0.5  # Columns get less weight than tables
                            
                            # Boost for common column patterns
                            if comp in {'id', 'name', 'date', 'time', 'amount', 'total', 'price', 'qty'}:
                                weights[comp] += 0.5
                            elif comp.endswith('_id'):
                                weights[comp[:-3]] += 0.3  # Reference to another table
        
        # 3. Analyze relationships
        for rel in self.schema_dict['relationships']:
            from_table = rel['from'].split('.')[1].lower()
            to_table = rel['to'].split('.')[1].lower()
            
            # Boost terms that appear in relationships
            weights[from_table] += 0.7
            weights[to_table] += 0.7
        
        # Normalize weights to reasonable range
        max_weight = max(weights.values()) if weights else 1.0
        return {k: (v/max_weight)*1.5 + 0.5 for k, v in weights.items()}  # Range: 0.5-2.0
    
    def update_weights_from_feedback(self, query: str, correct_tables: List[str]):
        """Dynamically update weights based on successful feedback"""
        doc = nlp(query.lower())
        
        # Extract entities from query
        for chunk in doc.noun_chunks:
            entity = chunk.text.lower()
            # Boost weight for entities that led to correct tables
            if entity in self.entity_weights:
                self.entity_weights[entity] = min(2.0, self.entity_weights[entity] * 1.2)
            else:
                self.entity_weights[entity] = 1.2  # Initial weight for new entities
        
        # Also learn from table names themselves
        for table in correct_tables:
            schema, table_name = table.split('.')
            for term in table_name.lower().split('_'):
                if term in self.entity_weights:
                    self.entity_weights[term] = min(2.0, self.entity_weights[term] * 1.1)
                else:
                    self.entity_weights[term] = 1.1
        
        # Save updated weights
        with open(self.weights_file, 'w') as f:
            json.dump(self.entity_weights, f, indent=2)
    
    def identify_tables(self, query: str) -> Tuple[List[str], bool]:
        """
        Identify relevant tables for a query
        Returns tuple of (identified_tables, should_display)
        """
        # First check for similar feedback
        feedback = self.feedback_learner.get_similar_feedback(query)
        if feedback:
            print("\n[System] Found matching feedback:")
            for i, item in enumerate(feedback[:3], 1):
                print(f"{i}. Similarity: {item['similarity']:.2f} ({item.get('type', 'unknown')})")
                if item.get('pattern'):
                    print(f"   Pattern: {item['pattern']}")
                print(f"   Query: {item['query']}")
                print(f"   Tables: {', '.join(item['tables'])}")
                if item.get('count'):
                    print(f"   Count: {item['count']}")
            
            tables = self._select_best_tables(feedback)
            return tables, True  # Always display feedback-based results
        
        # NLP-based identification
        tables = self._identify_tables_nlp(query)
        total_weight = sum(self._calculate_query_weights(query))
        
        # Decide whether to display based on confidence
        should_display = total_weight >= self.MIN_WEIGHT_TO_DISPLAY
        return tables, should_display
    
    def _calculate_query_weights(self, query: str) -> List[float]:
        """Calculate weights for all entities in query"""
        doc = nlp(query.lower())
        return [
            self.entity_weights[chunk.text.lower()] 
            for chunk in doc.noun_chunks 
            if chunk.text.lower() in self.entity_weights
        ]
    
    def _select_best_tables(self, feedback: List[Dict]) -> List[str]:
        """Select best tables from similar feedback using weighted scoring"""
        table_scores = defaultdict(float)
        
        for item in feedback:
            # Different weight based on match type
            if item.get('type') == 'exact':
                weight = 2.0
            elif item.get('type') == 'pattern':
                weight = 1.5 * (item.get('count', 1) ** 0.5)  # Favor patterns with more examples
            else:  # semantic
                weight = item['similarity'] ** 2
                
            for table in item['tables']:
                table_scores[table] += weight
                
        # Get top tables by score (minimum score of 1.0)
        sorted_tables = sorted(table_scores.items(), key=lambda x: -x[1])
        return [t[0] for t in sorted_tables if t[1] >= 1.0][:5]  # Return max 5 tables
    
    def _identify_tables_nlp(self, query: str) -> List[str]:
        """Identify tables using NLP analysis of the query"""
        doc = nlp(query.lower())
        entities = self._extract_entities(doc)
        tables = self._match_entities_to_tables(entities)
        tables = self._expand_with_relationships(tables)
        
        return tables
    
    def _extract_entities(self, doc) -> Dict[str, float]:
        """Extract relevant entities from the query with weights"""
        entities = defaultdict(float)
        skip_next = False
        
        for i, token in enumerate(doc):
            if skip_next:
                skip_next = False
                continue
                
            if token.text in ('=', '!=') and i > 0:
                # Found a condition, mark the pattern
                entities[f"{doc[i-1].lemma_}=[CONDITION]"] = 1.5  # Boost condition patterns
                skip_next = True  # Skip the value
            elif token.like_num:
                entities['[YEAR]'] = 1.0
            elif token.is_quote:
                entities['[QUOTED]'] = 1.0
            else:
                entities[token.lemma_] = self.entity_weights.get(token.lemma_, 1.0)
        
        return entities
    
    def _match_entities_to_tables(self, entities: Dict[str, float]) -> List[str]:
        """Case-insensitive table matching with proper schema handling"""
        table_scores = defaultdict(float)
        has_conditions = any('=' in e for e in entities)
        
        for schema in self.schema_dict['tables']:
            for table in self.schema_dict['tables'][schema]:
                full_name = f"{schema}.{table}"
                table_lower = full_name.lower()
                
                # Bonus for tables likely to contain conditions
                if has_conditions and any(kw in table_lower 
                                       for kw in ['detail', 'info', 'data']):
                    table_scores[full_name] += 1.2
                
                # Match against each entity
                for entity, weight in entities.items():
                    # Handle condition patterns (column=[CONDITION])
                    if '=' in entity:
                        col_part = entity.split('=')[0]
                        if col_part in table_lower:
                            table_scores[full_name] += weight * 2.0
                    # Regular matching
                    elif entity in table_lower:
                        table_scores[full_name] += weight * 1.5
        
        # Return tables sorted by score (highest first)
        return sorted(
            [t for t, s in table_scores.items() if s >= self.MIN_WEIGHT_TO_ACCEPT],
            key=lambda x: -table_scores[x]
        )
    
    def _expand_with_relationships(self, tables: List[str]) -> List[str]:
        """Expand table list based on foreign key relationships"""
        expanded_tables = set(tables)
        
        for rel in self.schema_dict["relationships"]:
            from_parts = rel["from"].split('.')
            to_parts = rel["to"].split('.')
            
            from_table = f"{from_parts[0]}.{from_parts[1]}"
            to_table = f"{to_parts[0]}.{to_parts[1]}"
            
            # If one side is in our tables, add the other side
            if from_table in expanded_tables and to_table not in expanded_tables:
                expanded_tables.add(to_table)
            elif to_table in expanded_tables and from_table not in expanded_tables:
                expanded_tables.add(from_table)
        
        return list(expanded_tables)
    
    def build_cross_schema_index(self) -> Dict[str, List[str]]:
        """Build index of columns that appear in multiple schemas"""
        column_map = defaultdict(list)
        
        for schema_name, tables in self.schema_dict["columns"].items():
            for table_name, columns in tables.items():
                for column_name in columns:
                    full_name = f"{schema_name}.{table_name}.{column_name}"
                    column_map[column_name.lower()].append(full_name)
        
        return {k: v for k, v in column_map.items() if len(v) > 1}

if __name__ == "__main__":
    analyzer = DatabaseAnalyzer()
    analyzer.run()