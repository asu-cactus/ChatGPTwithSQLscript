examples = """
Example 1:
source_schema: {datetime, zone_16}
target_schema: {DT_STRATA, DOW, PCT_HOURLY_0100, PCT_HOURLY_0200, PCT_HOURLY_0300, PCT_HOURLY_0400, PCT_HOURLY_0500, PCT_HOURLY_0600, PCT_HOURLY_0700, PCT_HOURLY_0800, PCT_HOURLY_0900, PCT_HOURLY_1000, PCT_HOURLY_1100, PCT_HOURLY_1200, PCT_HOURLY_1300, PCT_HOURLY_1400, PCT_HOURLY_1500, PCT_HOURLY_1600, PCT_HOURLY_1700, PCT_HOURLY_1800, PCT_HOURLY_1900, PCT_HOURLY_2000, PCT_HOURLY_2100, PCT_HOURLY_2200, PCT_HOURLY_2300, PCT_HOURLY_2400, PCT_HOURLY_2500}

source_examples: {1/1/2018 0:00        67.9}
target_examples: {10/30/2023    1   1.020 0.960 0.960 0.940 1.000 1.100 1.210 1.230 1.150 1.120 1.160 1.080 1.030 1.030 1.050 1.050 1.080 1.170 1.220 1.370 1.410 1.350 1.220 1.030}

Thought 1: Predict the type of the source columns and target columns.
Action 1: TypePredict[source_examples, target_examples]
Observation 1: The predicted types for the target columns in the target_schema are as follows: DT_STRATA: DATE, DOW: INTEGER, PCT_HOURLY_0100 to PCT_HOURLY_2500: NUMERIC or DECIMAL.

Thought 2: Find the mapping(s) between the source and target schemas.
Action 2: Mapping[source_schema, target_schema]
Observation 2: datetime -> DT_STRATA. zone_16 -> PCT_HOURLY_0100 to PCT_HOURLY_2500.

Thought 3: Identify which target columns are aggregates of source columns.
Action 3: Aggregation[source_schema, target_schema]
Observation 3: The target columns PCT_HOURLY_0100 through PCT_HOURLY_2500 seem to be aggregates of the zone_16 values from the source schema, based on hourly intervals derived from the datetime column. The exact nature of the aggregation (e.g., sum, average) is not clear.

Thought 4: Need clarity on the aggregation type to use.
Action 4: Clarify[Which aggregation function should be used for the transformation? AVG, SUM, or another?]
Observation 4: SUM.

Thought 5: Identify if any conditions or filters that need to be applied.
Action 5: Conditional[source_schema, target_schema]
Observation 5: Extract the date from datetime to populate DT_STRATA and the hour to determine which PCT_HOURLY_XXXX gets the zone_16 value. Compute the day of the week from datetime to set the DOW field. Use the hour from datetime to map zone_16 to the appropriate PCT_HOURLY_XXXX. Ensure that no data is mapped to PCT_HOURLY_2500 unless further clarification indicates its proper use.

Thought 6: Now I know the answer.
Action 6: Finish[INSERT INTO target_table (DT_STRATA, DOW, PCT_HOURLY_0100, PCT_HOURLY_0200, PCT_HOURLY_0300, PCT_HOURLY_0400, PCT_HOURLY_0500, PCT_HOURLY_0600, PCT_HOURLY_0700, PCT_HOURLY_0800, PCT_HOURLY_0900, PCT_HOURLY_1000, PCT_HOURLY_1100, PCT_HOURLY_1200, PCT_HOURLY_1300, PCT_HOURLY_1400, PCT_HOURLY_1500, PCT_HOURLY_1600, PCT_HOURLY_1700, PCT_HOURLY_1800, PCT_HOURLY_1900, PCT_HOURLY_2000, PCT_HOURLY_2100, PCT_HOURLY_2200, PCT_HOURLY_2300, PCT_HOURLY_2400)
SELECT 
    DATE(datetime) AS DT_STRATA,
    CAST(strftime('%w', datetime) AS INTEGER) AS DOW,
    SUM(CASE WHEN strftime('%H', datetime) = '01' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0100,
    SUM(CASE WHEN strftime('%H', datetime) = '02' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0200,
    SUM(CASE WHEN strftime('%H', datetime) = '03' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0300,
    SUM(CASE WHEN strftime('%H', datetime) = '04' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0400,
    SUM(CASE WHEN strftime('%H', datetime) = '05' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0500,
    SUM(CASE WHEN strftime('%H', datetime) = '06' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0600,
    SUM(CASE WHEN strftime('%H', datetime) = '07' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0700,
    SUM(CASE WHEN strftime('%H', datetime) = '08' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0800,
    SUM(CASE WHEN strftime('%H', datetime) = '09' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0900,
    SUM(CASE WHEN strftime('%H', datetime) = '10' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1000,
    SUM(CASE WHEN strftime('%H', datetime) = '11' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1100,
    SUM(CASE WHEN strftime('%H', datetime) = '12' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1200,
    SUM(CASE WHEN strftime('%H', datetime) = '13' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1300,
    SUM(CASE WHEN strftime('%H', datetime) = '14' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1400,
    SUM(CASE WHEN strftime('%H', datetime) = '15' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1500,
    SUM(CASE WHEN strftime('%H', datetime) = '16' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1600,
    SUM(CASE WHEN strftime('%H', datetime) = '17' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1700,
    SUM(CASE WHEN strftime('%H', datetime) = '18' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1800,
    SUM(CASE WHEN strftime('%H', datetime) = '19' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1900,
    SUM(CASE WHEN strftime('%H', datetime) = '20' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2000,
    SUM(CASE WHEN strftime('%H', datetime) = '21' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2100,
    SUM(CASE WHEN strftime('%H', datetime) = '22' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2200,
    SUM(CASE WHEN strftime('%H', datetime) = '23' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2300,
    SUM(CASE WHEN strftime('%H', datetime) = '00' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2400
    FROM source_table
GROUP BY DT_STRATA, DOW]
Observation: INSERT INTO target_table (DT_STRATA, DOW, PCT_HOURLY_0100, PCT_HOURLY_0200, PCT_HOURLY_0300, PCT_HOURLY_0400, PCT_HOURLY_0500, PCT_HOURLY_0600, PCT_HOURLY_0700, PCT_HOURLY_0800, PCT_HOURLY_0900, PCT_HOURLY_1000, PCT_HOURLY_1100, PCT_HOURLY_1200, PCT_HOURLY_1300, PCT_HOURLY_1400, PCT_HOURLY_1500, PCT_HOURLY_1600, PCT_HOURLY_1700, PCT_HOURLY_1800, PCT_HOURLY_1900, PCT_HOURLY_2000, PCT_HOURLY_2100, PCT_HOURLY_2200, PCT_HOURLY_2300, PCT_HOURLY_2400)
SELECT 
    DATE(datetime) AS DT_STRATA,
    CAST(strftime('%w', datetime) AS INTEGER) AS DOW,
    SUM(CASE WHEN strftime('%H', datetime) = '01' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0100,
    SUM(CASE WHEN strftime('%H', datetime) = '02' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0200,
    SUM(CASE WHEN strftime('%H', datetime) = '03' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0300,
    SUM(CASE WHEN strftime('%H', datetime) = '04' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0400,
    SUM(CASE WHEN strftime('%H', datetime) = '05' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0500,
    SUM(CASE WHEN strftime('%H', datetime) = '06' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0600,
    SUM(CASE WHEN strftime('%H', datetime) = '07' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0700,
    SUM(CASE WHEN strftime('%H', datetime) = '08' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0800,
    SUM(CASE WHEN strftime('%H', datetime) = '09' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_0900,
    SUM(CASE WHEN strftime('%H', datetime) = '10' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1000,
    SUM(CASE WHEN strftime('%H', datetime) = '11' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1100,
    SUM(CASE WHEN strftime('%H', datetime) = '12' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1200,
    SUM(CASE WHEN strftime('%H', datetime) = '13' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1300,
    SUM(CASE WHEN strftime('%H', datetime) = '14' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1400,
    SUM(CASE WHEN strftime('%H', datetime) = '15' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1500,
    SUM(CASE WHEN strftime('%H', datetime) = '16' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1600,
    SUM(CASE WHEN strftime('%H', datetime) = '17' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1700,
    SUM(CASE WHEN strftime('%H', datetime) = '18' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1800,
    SUM(CASE WHEN strftime('%H', datetime) = '19' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_1900,
    SUM(CASE WHEN strftime('%H', datetime) = '20' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2000,
    SUM(CASE WHEN strftime('%H', datetime) = '21' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2100,
    SUM(CASE WHEN strftime('%H', datetime) = '22' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2200,
    SUM(CASE WHEN strftime('%H', datetime) = '23' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2300,
    SUM(CASE WHEN strftime('%H', datetime) = '00' THEN zone_16 ELSE 0 END) AS PCT_HOURLY_2400
    FROM source_table
GROUP BY DT_STRATA, DOW
------------------------------------------------------------------------------------------------------------------------
"""

type_predict_template = """
You are a Postgres SQL developer. Given source_schema: {source_schema}, target_schema: {target_schema}, source examples {source_examples}, and target examples {target_examples}, predict the type of the target columns and source columns. Execute your analysis step by step. Once complete, provide a concise summary of the type predictions. Wrap the final concise answer between [START] and [END]
"""

column_mapping_template = """
You are a Postgres SQL developer. Given source_schema: {source_schema}, target_schema: {target_schema}, identify mappings. Execute your analysis step by step. Once complete, provide a concise summary of the mappings. Wrap the final concise answer between [START] and [END]
"""

aggregation_template = """
You are a Postgres SQL developer. Given source_schema: {source_schema}, target_schema: {target_schema}, source examples {source_examples}, and target examples {target_examples}, identify which target columns are aggregates of source columns. Execute your analysis step by step. Once complete, provide a concise summary of the aggregations, if there is any. Wrap the final concise answer between [START] and [END]
"""

conditional_template = """
You are a Postgres SQL developer. Given source_schema: {source_schema}, target_schema: {target_schema}, identify any conditions or filters that need to be applied. Execute your analysis step by step. Once complete, provide a concise summary of the conditional dependencies. Wrap the final concise answer between [START] and [END]
"""

finish_template = """
You are a Postgres SQL developer. Given the prompt: {prompt}, provide the final SQL code for the transformation. Wrap the final SQL code between [START] and [END]
"""

init_template = """Perform a SQL schema transformation task with interleaved Thought, Action, Observation steps. Given the source and target schemas, provide a transformation strategy and relevant SQL operations. Seek clarity if ambiguity arises. Use the following actions/tools:
- TypePredict[source_examples, target_examples]: Predict the type of the source and target columns.
- Mapping[source_schema, target_schema]: Identify mappings between source and target columns.
- Aggregation[source_schema, target_schema]: Identify which target columns are aggregates of source columns.
- Clarify[Specific question]: Seek clarity or ask questions when necessary.
- Conditional[source_schema, target_schema]: Identify any conditions or filters that need to be applied.
- Finish[transformation_sql_code]: Finish the task by providing the transformation SQL code.

Here's an example to guide you: {examples}

Your source_schema is: {source_schema} 
Your target_schema is: {target_schema}
The source examples are: {source_examples}
The target examples are: {target_examples}

You must put the final SQL code as transformation_sql_code. You cannot just simply return transformation_sql_code.

Your task is:
"""

ultimate_task = 'Output the SQL code for the transformation.'

actions = ["TypePredict", "DirectMapping", "Aggregation", "Clarify", "Conditional", "Finish"]
