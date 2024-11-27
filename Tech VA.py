import streamlit as st
import pandas as pd
import thriftpy2
from collections import Counter
from collections import defaultdict


def count_top_entities_with_articles(df, entity_columns, top_n=10):
    """
    Extracts and counts the top entities detected in the dataframe and associates them with related articles.

    Parameters:
    - df: DataFrame containing the entity columns.
    - entity_columns: List of columns containing entities.
    - top_n: Number of top entities to return.

    Returns:
    - DataFrame with top entities, their counts, and related articles.
    """
    # Initialize a Counter object to keep track of entity counts
    entity_counter = Counter()
    # Dictionary to store the entities and related articles
    entity_articles = defaultdict(set)

    # Loop over the specified columns to count entities and store related articles
    for idx, row in df.iterrows():
        for col in entity_columns:
            entities = row[col] if pd.notna(row[col]) else ''
            for entity in entities.split(","):
                entity = entity.strip()
                if entity:
                    entity_counter[entity] += 1
                    entity_articles[entity].add(idx)  # Store the index of the article (or another identifier)

    # Convert the counter to a DataFrame
    top_entities = pd.DataFrame(entity_counter.most_common(top_n), columns=['Entity', 'Count'])
    top_entities['Related Articles'] = top_entities['Entity'].map(entity_articles)

    return top_entities

def query_entity_in_articles(df, entity_columns, entity):
    """
    Queries whether a specified entity is present in the specified columns and returns a DataFrame
    with the entity, its count, and the related articles.

    Parameters:
    - df: DataFrame containing the entity columns.
    - entity_columns: List of columns containing entities.
    - entity: The entity to search for.

    Returns:
    - DataFrame with the entity, its count, and related articles.
    """
    # Initialize count and a set to store related articles
    entity_count = 0
    related_articles = set()

    # Loop over the specified columns to find the entity and store related articles
    for idx, row in df.iterrows():
        for col in entity_columns:
            entities = row[col] if pd.notna(row[col]) else ''
            if entity in [e.strip() for e in entities.split(",")]:
                entity_count += 1
                related_articles.add(idx)  # Store the index of the article (or another identifier)

    # Construct the resulting DataFrame in the required format
    result_df = pd.DataFrame({
        'Entity': [entity],
        'Count': [entity_count],
        'Related Articles': [related_articles]
    })

    return result_df





# Load data and initialize parameters
df_exploded = pd.read_csv('exploded_data_with_date.csv')
df_exploded['pubdate'] = pd.to_datetime(df_exploded['pubdate'], errors='coerce')

weekly_sentiment_score_with_total_df = pd.read_csv('df_with_weekly_sentiment_score.csv')

entity_columns = ['PERSON', 'ORG', 'GPE', 'PRODUCT']  # Columns to extract entities from

# Extract top entities
top_entities_df = count_top_entities_with_articles(df_exploded, entity_columns)
representative_entity = pd.read_csv('representative_entities.csv')
# Import necessary libraries
import streamlit as st
import pandas as pd

# Assuming count_top_entities_with_articles is already defined and loads df_exploded
top_person = count_top_entities_with_articles(df_exploded, ['PERSON'], top_n=20)
top_org = count_top_entities_with_articles(df_exploded, ['ORG'], top_n=20)
top_gpe = count_top_entities_with_articles(df_exploded, ['GPE'], top_n=20)
top_product = count_top_entities_with_articles(df_exploded, ['PRODUCT'], top_n=20)

st.set_page_config(layout="wide")
# Display a website logo at the top
st.logo("web_logo.png", size = 'large')

col1, col2, col3 = st.columns([2, 4, 3])

# Left column for displaying top entities and search functionality
with col1:
    st.header("Top Entities by Category")
    entity_clicked = None

    # Search bar for finding a specific entity
    search_query = st.text_input("Search for an Entity")
    if search_query:
        # Query for the entity using query_entity_in_articles
        search_result = query_entity_in_articles(df_exploded, entity_columns, search_query)

        if not search_result.empty and search_result['Count'].iloc[0] > 0:
            # Display the search result as a button to allow viewing related articles
            if st.button(f"View results for {search_result['Entity'].iloc[0]}"):
                entity_clicked = search_result['Entity'].iloc[0]
        else:
            st.write("Entity not found")

    # Display top entities for each category in the left column with clickable buttons
    st.subheader("Top Product")
    for _, row in top_product[2:5].iterrows():
        if st.button(row['Entity']):
            entity_clicked = row['Entity']

    st.subheader("Top Location")
    for _, row in top_gpe[0:3].iterrows():
        if st.button(row['Entity']):
            entity_clicked = row['Entity']

    st.subheader("Top Organization")
    for _, row in top_org[0:3].iterrows():
        if st.button(row['Entity']):
            entity_clicked = row['Entity']

    st.subheader("Top People")
    for _, row in top_person[16:19].iterrows():
        if st.button(row['Entity']):
            entity_clicked = row['Entity']




# Right column for displaying related articles when an entity is clicked
with col2:
    st.header("Related Articles")
    # Date range filter (always available)
    min_date = pd.to_datetime(df_exploded['pubdate']).min()
    max_date = pd.to_datetime(df_exploded['pubdate']).max()
    start_date, end_date = st.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Sort by date option
    sort_order = st.selectbox("Sort by Date", ["Newest First", "Oldest First"])
    ascending = sort_order == "Oldest First"

    if entity_clicked:
        # Retrieve related articles for the clicked or searched entity
        temp = pd.concat([top_person, top_org, top_gpe, top_product]).loc[
            lambda x: x['Entity'] == entity_clicked, 'Related Articles'
        ]
        entity_sentiment = representative_entity[representative_entity['Entity'] == entity_clicked]['Sentiment Score']
        try:
            sentiment_value = round(entity_sentiment.values[0] * 100/0.279)  # Convert to percentage
            if (sentiment_value > 100):
                st.header(f"Sentiment Score of {entity_clicked}: :green[{sentiment_value}]")
            else:
                st.header(f"Sentiment Score of {entity_clicked}: :red[{sentiment_value}]")
            st.write("""Scores greater than 100 indicate positive sentiment, 
                        and scores below 100 indicate negative sentiment""")
        except IndexError:
            st.header(f"{entity_clicked}, Sentiment: Not Available")

        if temp.empty:
            # If the clicked entity came from a search, run the query again for related articles
            temp = query_entity_in_articles(df_exploded, entity_columns, entity_clicked)['Related Articles']

        if not temp.empty:
            related_articles_clicked = df_exploded.iloc[list(temp.values[0])]

            # Apply date filter and sort order to the related articles
            filtered_articles = related_articles_clicked[
                (related_articles_clicked['pubdate'] >= pd.to_datetime(start_date)) &
                (related_articles_clicked['pubdate'] <= pd.to_datetime(end_date))
            ].sort_values(by='pubdate', ascending=ascending)

            # Display title, cleaned_content, and summary for the filtered and sorted related articles
            for index, row in filtered_articles.iterrows():
                st.subheader(f"{row['title']}")
                st.write(f"Summary: {row['summary']}")
                st.write(f"Date: {row['pubdate']}")
                with st.expander("Show Content"):
                    st.write(f"{row['cleaned_content']}")
                st.write("---")
        else:
            st.write("No related articles found.")
    else:
        st.write("Please select or search an entity to view related articles.")

with col3:
    st.header("Sentiment Dashboard")

    if entity_clicked:
        # Filter the weekly sentiment data for the selected entity
        weekly_sentiment_score_with_total_df['Weekly Sentiment Score'] = weekly_sentiment_score_with_total_df['Weekly Sentiment Score']/0.279 * 100
        weekly_data = weekly_sentiment_score_with_total_df[
            weekly_sentiment_score_with_total_df['Entity'] == entity_clicked
            ]

        if not weekly_data.empty:
            # Line chart for weekly sentiment scores
            st.line_chart(
                weekly_data.set_index('Week')['Weekly Sentiment Score'],
                use_container_width=True
            )
            # Bar chart for the number of articles per week
            st.bar_chart(
                weekly_data.set_index('Week')['Number of Articles'],
                use_container_width=True
            )
        else:
            st.write("No sentiment data available for this entity.")
    else:
        st.write("Select or search an entity to view its sentiment dashboard.")






