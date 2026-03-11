# Conversational AI Business Intelligence Dashboard

A Streamlit-based business intelligence application that turns natural-language questions into interactive dashboards over uploaded CSV data.

The system uses:

- Streamlit for the UI
- SQLite as the queryable data layer
- Pandas for lightweight dataframe operations and exports
- Plotly for interactive charts
- Groq + Llama 3.1 8B Instant for compact query interpretation and short AI insights

## Features

- Upload your own CSV dataset
- Load uploaded data into an in-memory SQLite database
- Ask plain-English business questions
- Map queries to dataset fields safely
- Run SQL-backed aggregations for dashboard visuals
- Automatically recommend chart types
- Override chart type manually
- Show KPI cards and a short AI insight
- Support lightweight follow-up queries such as `Now show the same analysis as a pie chart`
- Export visual data and query interpretation as CSV
- Handle invalid model output and ambiguous prompts with safe fallbacks

## Architecture

The app follows this flow:

1. User uploads a CSV file
2. The CSV is loaded into SQLite
3. The user enters a natural-language query
4. The LLM interprets the query into:
   - grouping field
   - metric field
   - aggregation
   - chart type
5. The app validates the interpretation against the uploaded dataset schema
6. SQL aggregation runs on SQLite
7. Plotly renders the dashboard visuals
8. A short AI insight is generated from compact numeric summaries only

## Safety Design

To keep the system stable and compact:

- The LLM never receives raw dataframe rows
- The LLM never receives `dataframe.to_string()`
- The LLM only receives:
  - the current user query
  - compact schema labels
  - numeric summaries for the insight
- No conversation history is sent to the model
- `max_tokens` is capped in the LLM client
- JSON parsing is guarded with safe fallbacks
- Invalid columns and aggregation failures degrade gracefully

## Project Structure

- [app.py](/workspace/app.py) or local `app.py`: main Streamlit app, SQL workflow, dashboard UI
- [llm_engine.py](/workspace/llm_engine.py) or local `llm_engine.py`: Groq request handling, query interpretation, short insights
- [charts.py](/workspace/charts.py) or local `charts.py`: Plotly chart rendering
- `requirements.txt`: Python dependencies
- `.gitignore`: ignored local, cache, and secret files

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

## Example Queries

The app generates schema-aware prompts based on the uploaded dataset, but common examples are:

- `Show revenue by region`
- `Show average profit by category`
- `Show count by customer segment`
- `Which product line has the highest sales?`
- `Now show the same analysis as a pie chart`
- `Instead, make it average by region`

## Hackathon Demo Tips

For a strong demo, show:

1. Upload of a CSV file
2. A first query that creates a grouped business chart
3. A second query that changes the metric
4. A follow-up query that changes only the chart style or aggregation
5. The architecture section and query interpretation panel
6. Export of dashboard data

## Evaluation Mapping

### Accuracy

- Query interpretation is schema-validated
- SQL aggregations run on the uploaded dataset
- Ambiguous queries fall back safely
- Query interpretation is visible in the UI

### Aesthetics and UX

- Power BI-style report layout
- Interactive Plotly visuals
- Filters, KPIs, loading states, and exports

### Approach and Innovation

- Clear `Prompt -> LLM -> SQLite -> Visualization` pipeline
- Compact prompt design
- Hallucination-resistant field validation
- Follow-up query support without sending full conversation history

## Notes

- The database is SQLite in memory for each uploaded file.
- This design is ideal for hackathon demos and medium-size datasets.
- For production-scale deployment, the same architecture can be adapted to PostgreSQL, DuckDB, or a warehouse.
