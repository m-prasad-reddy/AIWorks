# analysis/table_identifier.py
import os
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import spacy

nlp = spacy.load("en_core_web_sm")

class TableIdentifier:
    MIN_WEIGHT_TO_DISPLAY = 0.8
    MIN_WEIGHT_TO_ACCEPT = 1.0
    
    def __init__(self, schema_dict: Dict, feedback_manager):
        self.schema_dict = schema_dict
        self.feedback_manager = feedback_manager
        self.cross_schema_index = self.build_cross_schema_index()
        self.weights_file = os.path.join("schema_cache", schema_dict.get('database', 'default'), "weights.json")
        self.entity_weights = self._initialize_weights()
    
    def identify_tables(self, query: str) -> Tuple[List[str], bool]:
        feedback = self.feedback_manager.get_similar_feedback(query)
        if feedback:
            self._display_feedback(feedback)
            tables = self._select_best_tables(feedback)
            return tables, True
        
        tables = self._identify_tables_nlp(query)
        total_weight = sum(self._calculate_query_weights(query))
        return tables, total_weight >= self.MIN_WEIGHT_TO_DISPLAY
    
    def update_weights_from_feedback(self, query: str, correct_tables: List[str]):
        doc = nlp(query.lower())
        
        for chunk in doc.noun_chunks:
            entity = chunk.text.lower()
            if entity in self.entity_weights:
                self.entity_weights[entity] = min(2.0, self.entity_weights[entity] * 1.2)
            else:
                self.entity_weights[entity] = 1.2
        
        for table in correct_tables:
            schema, table_name = table.split('.')
            for term in table_name.lower().split('_'):
                if term in self.entity_weights:
                    self.entity_weights[term] = min(2.0, self.entity_weights[term] * 1.1)
                else:
                    self.entity_weights[term] = 1.1
        
        with open(self.weights_file, 'w') as f:
            json.dump(self.entity_weights, f, indent=2)
    
    def _initialize_weights(self) -> Dict[str, float]:
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file) as f:
                    return json.load(f)
            except Exception:
                pass
        
        weights = self._generate_base_weights()
        os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
        with open(self.weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
        return weights
    
    def _generate_base_weights(self) -> Dict[str, float]:
        weights = defaultdict(float)
        
        for schema, tables in self.schema_dict['tables'].items():
            for table_name in tables.keys():
                components = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[a-z]+', table_name.lower())
                for comp in components:
                    if len(comp) > 2:
                        weights[comp] += 1.0
                        if comp.endswith(('s', 'es')):
                            weights[comp[:-1]] += 0.5
                        if comp in {'tbl', 'table', 'rel', 'xref', 'map', 'link'}:
                            weights[comp] += 0.3
        
        for schema, tables in self.schema_dict['columns'].items():
            for table_name, columns in tables.items():
                for column_name in columns.keys():
                    col_components = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[a-z]+', column_name.lower())
                    for comp in col_components:
                        if len(comp) > 2:
                            weights[comp] += 0.5
                            if comp in {'id', 'name', 'date', 'time', 'amount', 'total', 'price', 'qty'}:
                                weights[comp] += 0.5
                            elif comp.endswith('_id'):
                                weights[comp[:-3]] += 0.3
        
        for rel in self.schema_dict['relationships']:
            from_table = rel['from'].split('.')[1].lower()
            to_table = rel['to'].split('.')[1].lower()
            weights[from_table] += 0.7
            weights[to_table] += 0.7
        
        max_weight = max(weights.values()) if weights else 1.0
        return {k: (v/max_weight)*1.5 + 0.5 for k, v in weights.items()}
    
    def _calculate_query_weights(self, query: str) -> List[float]:
        doc = nlp(query.lower())
        return [
            self.entity_weights[chunk.text.lower()] 
            for chunk in doc.noun_chunks 
            if chunk.text.lower() in self.entity_weights
        ]
    
    def _select_best_tables(self, feedback: List[Dict]) -> List[str]:
        table_scores = defaultdict(float)
        
        for item in feedback:
            if item.get('type') == 'exact':
                weight = 2.0
            elif item.get('type') == 'pattern':
                weight = 1.5 * (item.get('count', 1) ** 0.5)
            else:
                weight = item['similarity'] ** 2
                
            for table in item['tables']:
                table_scores[table] += weight
                
        sorted_tables = sorted(table_scores.items(), key=lambda x: -x[1])
        return [t[0] for t in sorted_tables if t[1] >= 1.0][:5]
    
    def _display_feedback(self, feedback: List[Dict]):
        print("\n[System] Found matching feedback:")
        for i, item in enumerate(feedback[:3], 1):
            print(f"{i}. Similarity: {item['similarity']:.2f} ({item.get('type', 'unknown')})")
            if item.get('pattern'):
                print(f"   Pattern: {item['pattern']}")
            print(f"   Query: {item['query']}")
            print(f"   Tables: {', '.join(item['tables'])}")
            if item.get('count'):
                print(f"   Count: {item['count']}")
    
    def _identify_tables_nlp(self, query: str) -> List[str]:
        doc = nlp(query.lower())
        entities = self._extract_entities(doc)
        tables = self._match_entities_to_tables(entities)
        return self._expand_with_relationships(tables)
    
    def _extract_entities(self, doc) -> Dict[str, float]:
        entities = defaultdict(float)
        skip_next = False
        
        for i, token in enumerate(doc):
            if skip_next:
                skip_next = False
                continue
                
            if token.text in ('=', '!=') and i > 0:
                entities[f"{doc[i-1].lemma_}=[CONDITION]"] = 1.5
                skip_next = True
            elif token.like_num:
                entities['[YEAR]'] = 1.0
            elif token.is_quote:
                entities['[QUOTED]'] = 1.0
            else:
                entities[token.lemma_] = self.entity_weights.get(token.lemma_, 1.0)
        
        return entities
    
    def _match_entities_to_tables(self, entities: Dict[str, float]) -> List[str]:
        table_scores = defaultdict(float)
        has_conditions = any('=' in e for e in entities)
        
        for schema in self.schema_dict['tables']:
            for table in self.schema_dict['tables'][schema]:
                full_name = f"{schema}.{table}"
                table_lower = full_name.lower()
                
                if has_conditions and any(kw in table_lower for kw in ['detail', 'info', 'data']):
                    table_scores[full_name] += 1.2
                
                for entity, weight in entities.items():
                    if '=' in entity:
                        col_part = entity.split('=')[0]
                        if col_part in table_lower:
                            table_scores[full_name] += weight * 2.0
                    elif entity in table_lower:
                        table_scores[full_name] += weight * 1.5
        
        return sorted(
            [t for t, s in table_scores.items() if s >= self.MIN_WEIGHT_TO_ACCEPT],
            key=lambda x: -table_scores[x]
        )
    
    def _expand_with_relationships(self, tables: List[str]) -> List[str]:
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
    
    def build_cross_schema_index(self) -> Dict[str, List[str]]:
        column_map = defaultdict(list)
        
        for schema_name, tables in self.schema_dict["columns"].items():
            for table_name, columns in tables.items():
                for column_name in columns:
                    full_name = f"{schema_name}.{table_name}.{column_name}"
                    column_map[column_name.lower()].append(full_name)
        
        return {k: v for k, v in column_map.items() if len(v) > 1}