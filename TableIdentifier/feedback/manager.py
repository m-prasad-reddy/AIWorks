# feedback/manager.py
import os
import json
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")

class FeedbackManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.feedback_dir = os.path.join("feedback", db_name)
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.feedback_cache = {}
        self.pattern_cache = {}
        self._load_feedback_cache()
    
    def store_feedback(self, query: str, correct_tables: List[str], schema_dict: Dict) -> bool:
        valid_tables, invalid_tables = self.validate_tables(correct_tables, schema_dict)
        
        if invalid_tables:
            print(f"\n[Error] These tables don't exist: {', '.join(invalid_tables)}")
            return False
            
        normalized_tables = [t.lower() for t in valid_tables]
        existing = self._find_exact_match(query)
        
        if existing:
            self._update_feedback(existing, normalized_tables)
            print("\n[System] Updated existing feedback record")
        else:
            self._create_new_feedback(query, normalized_tables)
            print("\n[System] New feedback stored")
        
        self._load_feedback_cache()
        return True
    
    def validate_tables(self, tables: List[str], schema_dict: Dict) -> Tuple[List[str], List[str]]:
        valid_tables = []
        invalid_tables = []
        
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
            
            if schema_lower not in schema_map:
                invalid_tables.append(table)
                continue
                
            actual_schema = schema_map[schema_lower]
            
            if table_lower not in table_maps[actual_schema]:
                invalid_tables.append(table)
                continue
                
            actual_table = table_maps[actual_schema][table_lower]
            valid_tables.append(f"{actual_schema}.{actual_table}")
                
        return valid_tables, invalid_tables
    
    def get_similar_feedback(self, query: str, threshold: float = 0.85) -> Optional[List[Dict]]:
        query_lower = query.lower()
        if query_lower in self.feedback_cache:
            return [{
                'similarity': 1.0,
                'query': query,
                'tables': self.feedback_cache[query_lower]['tables'],
                'timestamp': self.feedback_cache[query_lower]['timestamp'],
                'type': 'exact'
            }]

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
        
        feedback_items.sort(key=lambda x: x["similarity"], reverse=True)
        return feedback_items if feedback_items else None
    
    def _load_feedback_cache(self):
        self.feedback_cache.clear()
        self.pattern_cache.clear()
        
        for fname in os.listdir(self.feedback_dir):
            if fname.endswith("_meta.json"):
                with open(os.path.join(self.feedback_dir, fname)) as f:
                    meta = json.load(f)
                    normalized_tables = [t.lower() for t in meta['tables']]
                    
                    self.feedback_cache[meta['query'].lower()] = {
                        'tables': normalized_tables,
                        'timestamp': meta['timestamp']
                    }
                    
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
        doc = nlp(query.lower())
        pattern = []
        skip_next = False
        
        for i, token in enumerate(doc):
            if skip_next:
                skip_next = False
                continue
                
            if token.text in ('=', '!=') and i > 0:
                pattern.append(f"{doc[i-1].lemma_}=[CONDITION]")
                skip_next = True
            elif token.like_num:
                pattern.append('[YEAR]')
            elif token.is_quote:
                pattern.append('[QUOTED]')
            else:
                pattern.append(token.lemma_)
        
        return ' '.join(pattern)
    
    def _normalize_query(self, query: str) -> str:
        query = re.sub(r'(\w+)\s*[=!]=\s*[\'"]?([\w\s]+)[\'"]?', 
                      r'\1=[CONDITION]', query, flags=re.IGNORECASE)
        query = ' '.join([word for word in query.split() if not word.isdigit()])
        return re.sub(r'[^a-z0-9\s]', '', query.lower())
    
    def _find_exact_match(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        for fname in os.listdir(self.feedback_dir):
            if fname.endswith("_meta.json"):
                with open(os.path.join(self.feedback_dir, fname)) as f:
                    meta = json.load(f)
                    if meta['query'].lower() == query_lower:
                        return fname.replace("_meta.json", "")
        return None
    
    def _update_feedback(self, feedback_id: str, tables: List[str]):
        meta_path = os.path.join(self.feedback_dir, f"{feedback_id}_meta.json")
        with open(meta_path, 'r+') as f:
            meta = json.load(f)
            meta['tables'] = tables
            meta['timestamp'] = datetime.now().isoformat()
            f.seek(0)
            json.dump(meta, f)
            f.truncate()
    
    def _create_new_feedback(self, query: str, tables: List[str]):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        embedding = self.model.encode(self._normalize_query(query))
        
        np.save(os.path.join(self.feedback_dir, f"{timestamp}_emb.npy"), embedding)
        with open(os.path.join(self.feedback_dir, f"{timestamp}_meta.json"), 'w') as f:
            json.dump({
                'query': query,
                'tables': tables,
                'timestamp': datetime.now().isoformat()
            }, f)