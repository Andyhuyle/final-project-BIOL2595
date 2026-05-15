library(dplyr)
library(tidyr)

# -----------------------------
# Load data
# -----------------------------
cases <- read.csv("pc_cases.csv") %>% mutate(group = "Case")
controls <- read.csv("pc_controls.csv") %>% mutate(group = "Control")

df <- bind_rows(cases, controls)

# =========================================================
# HELPER FUNCTION: make Table 1 block
# =========================================================
make_block <- function(data, var_name) {

  data %>%
    group_by(.data[[var_name]], group) %>%
    summarise(n = n(), .groups = "drop") %>%
    pivot_wider(names_from = group, values_from = n, values_fill = 0) %>%
    mutate(
      Total = Case + Control,
      variable = var_name,
      level = .data[[var_name]]
    ) %>%
    select(variable, level, Case, Control, Total)
}

# =========================================================
# 1. AGE BINS
# =========================================================
df <- df %>%
  mutate(
    age_bin = cut(
      anchor_age,
      breaks = seq(0, 100, by = 10),
      right = FALSE,
      include.lowest = TRUE
    )
  )

age_table <- make_block(df, "age_bin")

# =========================================================
# 2. RACE
# =========================================================
race_table <- make_block(df, "race")

# =========================================================
# 3. DOD (Alive vs Dead)
# =========================================================
df <- df %>%
  mutate(death = ifelse(is.na(dod), "Alive", "Dead"))

dod_table <- make_block(df, "death")

# =========================================================
# 4. COMBINE EVERYTHING
# =========================================================
table1 <- bind_rows(
  age_table,
  race_table,
  dod_table
)

# =========================================================
# PRINT
# =========================================================
print(table1)

# =========================================================
# SAVE
# =========================================================
write.csv(table1, "Table1_final.csv", row.names = FALSE)

cat("Saved Table1_final.csv\n")