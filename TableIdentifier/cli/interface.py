# cli/interface.py
import os
from typing import Dict, List, Optional

class DatabaseAnalyzerCLI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def run(self):
        print("=== Database Schema Analyzer ===")
        while True:
            print("\nMain Menu:")
            print("1. Load Database Configuration")
            print("2. Exit")
            
            choice = input("\nEnter your choice (1-2): ").strip()
            
            if choice == "1":
                self._handle_database_config()
                if self.analyzer.connection_manager.is_connected():  # Updated this line
                    self._main_analysis_loop()
            elif choice == "2":
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _handle_database_config(self):
        config_path = input("\nEnter config file path [default: database_configurations.json]: ").strip()
        if not config_path:
            config_path = os.path.join(os.getcwd(), "database_configurations.json")
        
        try:
            configs = self.analyzer.load_configs(config_path)
            self._display_config_options(configs)
        except Exception as e:
            print(f"Error loading configurations: {str(e)}")
    
    def _display_config_options(self, configs: Dict):
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
                    self.analyzer.set_current_config(configs[config_key])
                    self.analyzer.connect_to_database()
                    return
                elif choice_idx == len(configs):
                    return
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def _main_analysis_loop(self):
        while True:
            print("\nOptions:")
            print("1. Enter query mode")
            print("2. Switch database configuration")
            print("3. Back to main menu")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                self._query_mode()
            elif choice == "2":
                self._handle_database_config()
                if not self.analyzer.connection_manager.is_connected():  # Updated this line
                    return
            elif choice == "3":
                self.analyzer.close_connection()
                return
            else:
                print("Invalid choice. Please try again.")
    
    def _query_mode(self):
        while True:
            query = input("\nEnter your database query (or 'back' to return to options): ").strip()
            if query.lower() == 'back':
                return
            
            if not query:
                print("No query entered.")
                continue
            
            print("\nAnalyzing query...")
            self.analyzer.process_query(query)