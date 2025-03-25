from dateutil import parser as date_parser

class DateHandler:
    def __init__(self):
        """Initialize DateHandler with default settings for SQL Server."""
        self.default_date = date_parser.parse('2000-01-01')  # Default for partial parsing
        # SQL Server format specifiers
        self.date_format = 23    # 'yyyy-mm-dd' for DATE
        self.datetime_format = 120  # 'yyyy-mm-dd hh:mi:ss' for DATETIME

    def parse_date_entities(self, question, nlp):
        """Extract date entities from the question using spaCy."""
        doc = nlp(question)
        dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
        return dates

    def _determine_column_type(self, column_type):
        """Map column type to appropriate SQL Server format."""
        if column_type == 'date':
            return self.date_format
        elif column_type in ['datetime', 'smalldatetime']:
            return self.datetime_format
        else:
            return self.date_format  # Default to DATE format for safety

    def generate_date_condition(self, date_str, date_column, column_type):
        """Generate SQL Server date conditions with CONVERT for a single date entity."""
        try:
            # Parse the date string with a default for partial inputs
            date_obj = date_parser.parse(date_str, default=self.default_date)
            date_str_lower = date_str.lower()
            sql_format = self._determine_column_type(column_type)

            # Handle relative dates (placeholder for future enhancement)
            if date_str_lower in ['last year', 'this year', 'next year']:
                return ''  # Could use GETDATE() and DATEADD for relative dates

            # Year and month (e.g., "February 2023")
            if date_obj.year and date_obj.month and not date_obj.day:
                start_date = date_obj.replace(day=1)
                if date_obj.month == 12:
                    end_date = date_obj.replace(year=date_obj.year + 1, month=1, day=1)
                else:
                    end_date = date_obj.replace(month=date_obj.month + 1, day=1)
                if column_type == 'date':
                    return (f"{date_column} >= CONVERT(DATE, '{start_date.strftime('%Y-%m-%d')}', {sql_format}) "
                            f"AND {date_column} < CONVERT(DATE, '{end_date.strftime('%Y-%m-%d')}', {sql_format})")
                else:  # datetime
                    return (f"{date_column} >= CONVERT(DATETIME, '{start_date.strftime('%Y-%m-%d')} 00:00:00', {sql_format}) "
                            f"AND {date_column} < CONVERT(DATETIME, '{end_date.strftime('%Y-%m-%d')} 00:00:00', {sql_format})")

            # Year only (e.g., "2023")
            elif date_obj.year and not date_obj.month:
                start_date = date_obj.replace(month=1, day=1)
                end_date = date_obj.replace(year=date_obj.year + 1, month=1, day=1)
                if column_type == 'date':
                    return (f"{date_column} >= CONVERT(DATE, '{start_date.strftime('%Y-%m-%d')}', {sql_format}) "
                            f"AND {date_column} < CONVERT(DATE, '{end_date.strftime('%Y-%m-%d')}', {sql_format})")
                else:  # datetime
                    return (f"{date_column} >= CONVERT(DATETIME, '{start_date.strftime('%Y-%m-%d')} 00:00:00', {sql_format}) "
                            f"AND {date_column} < CONVERT(DATETIME, '{end_date.strftime('%Y-%m-%d')} 00:00:00', {sql_format})")

            # Specific date (e.g., "2023-01-15")
            else:
                if column_type == 'date':
                    return f"{date_column} = CONVERT(DATE, '{date_obj.strftime('%Y-%m-%d')}', {sql_format})"
                else:  # datetime
                    return f"{date_column} = CONVERT(DATETIME, '{date_obj.strftime('%Y-%m-%d')} 00:00:00', {sql_format})"

        except ValueError:
            return ''  # Return empty string for unparseable dates

    def generate_conditions(self, question, date_column, column_type, nlp):
        """Generate all date conditions for a question."""
        date_entities = self.parse_date_entities(question, nlp)
        if not date_entities or not date_column:
            return ''
        conditions = [self.generate_date_condition(date_str, date_column, column_type) for date_str in date_entities]
        conditions = [cond for cond in conditions if cond]  # Filter out empty conditions
        return "WHERE " + " AND ".join(conditions) if conditions else ''
    