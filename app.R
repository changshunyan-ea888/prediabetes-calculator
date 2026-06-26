





library(shiny)
library(bslib)
library(ggplot2)
library(xgboost)
library(shapviz)
library(scales)






# 1. 读取已保存模型对象

obj <- readRDS("ifg_survival_xgb_shap_model.rds")

xgbmodel_surv <- obj$model
predictors    <- obj$predictors
time_var      <- obj$time_var
event_var     <- obj$event_var
ui_defaults   <- obj$ui_defaults
risk_cutoffs  <- obj$risk_cutoffs
basehaz_df    <- obj$basehaz_df


# 2. 工具函数

get_H0_t <- function(t, basehaz_df) {
  approx(
    x = basehaz_df$time,
    y = basehaz_df$hazard,
    xout = t,
    method = "linear",
    rule = 2
  )$y
}

predict_surv_risk <- function(newdata, horizon) {
  X_new <- data.matrix(newdata[, predictors, drop = FALSE])
  dnew  <- xgb.DMatrix(X_new)
  
  risk_score <- as.numeric(predict(xgbmodel_surv, newdata = dnew))
  H0_t <- get_H0_t(horizon, basehaz_df)
  
  surv_prob  <- exp(-H0_t * exp(risk_score))
  event_prob <- 1 - surv_prob
  
  list(
    risk_score = risk_score,
    surv_prob = surv_prob,
    event_prob = event_prob
  )
}

get_risk_group <- function(risk_score) {
  if (risk_score < risk_cutoffs$q1) {
    "Low risk"
  } else if (risk_score < risk_cutoffs$q2) {
    "Intermediate risk"
  } else {
    "High risk"
  }
}

get_risk_color <- function(group) {
  if (group == "Low risk") return("#2E86DE")
  if (group == "Intermediate risk") return("#F39C12")
  "#E74C3C"
}

fmt_pct <- function(x, digits = 1) {
  paste0(round(x * 100, digits), "%")
}

build_risk_curve_plot <- function(new_patient, pred_res_obj) {
  times <- seq(1, 5, by = 0.05)
  
  risks <- sapply(times, function(tt) {
    predict_surv_risk(new_patient, horizon = tt)$event_prob
  })
  
  plot_df <- data.frame(
    time = times,
    risk = risks
  )
  
  ggplot(plot_df, aes(x = time, y = risk)) +
    geom_line(linewidth = 1.4, color = "#ff4d4f") +
    geom_point(
      data = data.frame(
        time = c(3, 4, 5),
        risk = c(
          pred_res_obj$res_3$event_prob,
          pred_res_obj$res_4$event_prob,
          pred_res_obj$res_5$event_prob
        )
      ),
      aes(x = time, y = risk),
      size = 3,
      color = "#ff4d4f"
    ) +
    scale_x_continuous(breaks = 1:5, limits = c(1, 5)) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    theme_bw(base_size = 14) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#eceff5"),
      panel.border = element_rect(color = "#dfe4ee", fill = NA),
      axis.title = element_text(face = "bold"),
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(color = "#667085")
    ) +
    labs(
      title = "Predicted cumulative risk of FPG-defined incident prediabetes over time",
      subtitle = "Patient-specific time-dependent prediction",
      x = "Follow-up time (years)",
      y = "Cumulative risk"
    )
}

# 修改title

build_shap_plot <- function(shap_one) {
  sv_waterfall(
    shap_one,
    row_id = 1,
    fill_colors = c("#ff4d4f", "#2E86DE")
  ) +
    theme_bw(base_size = 13) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.border = element_rect(color = "#dfe4ee", fill = NA),
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(color = "#667085")
    ) +
    labs(
      title = "SHAP waterfall plot for the current patient",
      subtitle = "Positive contributions increase risk, while negative contributions decrease risk"
    )
}

# 修改title

build_report_html <- function(new_patient, pred_obj) {
  risk3 <- round(pred_obj$res_3$event_prob * 100, 1)
  risk4 <- round(pred_obj$res_4$event_prob * 100, 1)
  risk5 <- round(pred_obj$res_5$event_prob * 100, 1)
  score <- round(pred_obj$res_5$risk_score, 3)
  grp   <- get_risk_group(pred_obj$res_5$risk_score)
  
  html <- paste0(
    "<html><head><meta charset='UTF-8'><title>FPG-Defined Incident Prediabetes Risk Report</title>",
    "<style>",
    "body{font-family:Arial,Helvetica,sans-serif;margin:40px;color:#2f2f3a;line-height:1.7;}",
    "h1{color:#303241;} h2{color:#444;} ",
    "table{border-collapse:collapse;width:100%;margin-top:16px;margin-bottom:24px;}",
    "th,td{border:1px solid #dfe4ee;padding:10px;text-align:left;}",
    "th{background:#f4f6fb;} ",
    ".badge{display:inline-block;padding:6px 12px;border-radius:999px;color:#fff;font-weight:700;background:#ff4d4f;}",
    ".note{color:#667085;font-size:14px;}",
    "</style></head><body>",
    "<h1>FPG-Defined Incident Prediabetes Risk Prediction Report</h1>",
    "<p>This report is generated from a time-dependent model using five predictors.</p>",
    
    "<h2>Patient Inputs</h2>",
    "<table>",
    "<tr><th>Variable</th><th>Value</th></tr>",
    "<tr><td>Age</td><td>", new_patient$Age, "</td></tr>",
    "<tr><td>Body mass index</td><td>", new_patient$BMI, "</td></tr>",
    "<tr><td>Diastolic blood pressure</td><td>", new_patient$DBP, "</td></tr>",
    "<tr><td>Fasting plasma glucose</td><td>", new_patient$FPG, "</td></tr>",
    "<tr><td>Family history of diabetes</td><td>", new_patient$familyhistroyofdiabetes, "</td></tr>",
    "</table>",
    
    "<h2>Prediction Summary</h2>",
    "<p><b>Predicted risk score:</b> ", score, "</p>",
    "<p><b>Risk group:</b> <span class='badge'>", grp, "</span></p>",
    "<p><b>3-year risk of FPG-defined incident prediabetes:</b> ", risk3, "%</p>",
    "<p><b>4-year risk of FPG-defined incident prediabetes:</b> ", risk4, "%</p>",
    "<p><b>5-year risk of FPG-defined incident prediabetes:</b> ", risk5, "%</p>",
    
    "<h2>Interpretation</h2>",
    "<p>According to the time-dependent model, this patient is classified as <b>", grp,
    "</b>. The estimated cumulative risk of developing FPG-defined incident prediabetes is <b>", risk3,
    "%</b> at 3 years, <b>", risk4, "%</b> at 4 years, and <b>", risk5,
    "%</b> at 5 years.</p>",
    
    "<p class='note'>Note: This report is intended for research use and model demonstration. Clinical decisions should not rely solely on this output.</p>",
    "</body></html>"
  )
  
  html
}


# 3. 页面样式

custom_css <- "
:root {
  --main-red: #ff4d4f;
  --soft-red: #ff7875;
  --dark-text: #2f2f3a;
  --muted-text: #6b7280;
  --panel-bg: #f7f8fb;
  --card-bg: #ffffff;
  --line: #e8eaf1;
}
body {
  background: #f4f6fb;
  color: var(--dark-text);
  font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}
.main-title {
  font-size: 46px;
  font-weight: 800;
  line-height: 1.1;
  color: #303241;
  margin-bottom: 20px;
}
.subtitle-note {
  font-size: 15px;
  color: var(--muted-text);
  margin-bottom: 10px;
}
.side-card, .result-card, .plot-card, .summary-card, .download-card {
  background: var(--card-bg);
  border: 1px solid var(--line);
  border-radius: 18px;
  box-shadow: 0 4px 18px rgba(28, 39, 60, 0.06);
}
.side-card { padding: 20px 18px 20px 18px; }
.result-card, .plot-card, .summary-card, .download-card { padding: 18px; }
.metric-box {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 2px 10px rgba(28, 39, 60, 0.04);
  min-height: 120px;
}
.metric-label {
  font-size: 15px;
  font-weight: 600;
  color: #5b6474;
  margin-bottom: 8px;
}
.metric-value {
  font-size: 34px;
  font-weight: 800;
  color: var(--main-red);
  line-height: 1.1;
}
.metric-sub {
  margin-top: 8px;
  color: #7b8595;
  font-size: 13px;
}
.section-title {
  font-size: 22px;
  font-weight: 800;
  color: #303241;
  margin-bottom: 14px;
}
.shiny-input-container { width: 100% !important; }
.control-label {
  font-size: 16px;
  font-weight: 700;
  color: #3b4252;
}
.form-control, .selectize-input, .form-select {
  border-radius: 14px !important;
  min-height: 48px;
  font-size: 16px;
  border: 1px solid #dfe4ee !important;
  box-shadow: none !important;
}
.btn-predict {
  width: 100%;
  min-height: 52px;
  font-size: 18px;
  font-weight: 700;
  border-radius: 16px !important;
  border: none !important;
  background: linear-gradient(90deg, #ff4d4f 0%, #ff6b6b 100%) !important;
  color: white !important;
}
.btn-predict:hover { filter: brightness(0.98); }
.small-note {
  color: #7c8594;
  font-size: 13px;
}
.summary-text {
  font-size: 17px;
  line-height: 1.8;
  color: #424b5a;
}
.risk-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  color: white;
  font-size: 14px;
  font-weight: 700;
}
.hr-soft {
  border-top: 1px solid #edf0f5;
  margin: 14px 0 18px 0;
}
.plot-caption {
  font-size: 13px;
  color: #7b8595;
  margin-top: 8px;
}
.btn-download {
  width: 100%;
  margin-bottom: 10px;
  border-radius: 14px !important;
  min-height: 44px;
  font-weight: 700;
}
"


# 4. UI

ui <- page_fluid(
  theme = bs_theme(
    version = 5,
    bg = "#f4f6fb",
    fg = "#2f2f3a",
    primary = "#ff4d4f",
    base_font = font_google("Inter")
  ),
  tags$head(tags$style(HTML(custom_css))),
  
  fluidRow(
    column(
      width = 8,
      div(class = "main-title", "Risk calculator for FPG-defined incident prediabetes"),
      div(
        class = "subtitle-note",
        "Based on a time-dependent model for FPG-defined incident prediabetes with individualized SHAP explanation"
      )
    ),
    column(
      width = 4,
      div(
        class = "download-card",
        div(class = "section-title", "Downloads"),
        downloadButton("download_curve", "Download risk curve (PNG)", class = "btn btn-outline-danger btn-download"),
        downloadButton("download_shap", "Download SHAP plot (PNG)", class = "btn btn-outline-danger btn-download"),
        downloadButton("download_csv", "Download prediction results (CSV)", class = "btn btn-outline-danger btn-download"),
        downloadButton("download_report", "Download report (HTML)", class = "btn btn-outline-danger btn-download")
      )
    )
  ),
  
  fluidRow(
    column(
      width = 4,
      div(
        class = "side-card",
        div(class = "section-title", "Patient variables"),
        
        sliderInput(
          "Age", "Age:",
          min = ui_defaults$Age_min,
          max = ui_defaults$Age_max,
          value = ui_defaults$Age_default,
          step = 1
        ),
        
        numericInput(
          "BMI", "Body mass index:",
          value = ui_defaults$BMI_default,
          min = 10, max = 60, step = 0.1,
          width = "100%"
        ),
        
        numericInput(
          "DBP", "Diastolic blood pressure:",
          value = ui_defaults$DBP_default,
          min = 30, max = 150, step = 1,
          width = "100%"
        ),
        
        numericInput(
          "FPG", "Fasting plasma glucose:",
          value = ui_defaults$FPG_default,
          min = 2, max = 20, step = 0.01,
          width = "100%"
        ),
        
        selectInput(
          "familyhistroyofdiabetes",
          "Family history of diabetes:",
          choices = c("Negative (0)" = 0, "Positive (1)" = 1),
          selected = 0,
          width = "100%"
        ),
        
        br(),
        
        actionButton("predict_btn", "Predict", class = "btn-predict"),
        
        br(), br(),
        
        div(
          class = "small-note",
          "Predicted risk refers to FPG-defined incident prediabetes, defined by follow-up FPG 5.6 to <7.0 mmol/L among individuals with baseline FPG <5.6 mmol/L. Inputs should use the same variable definitions as in the training dataset."
        )
      )
    ),
    
    column(
      width = 8,
      
      fluidRow(
        column(
          width = 4,
          div(
            class = "metric-box",
            div(class = "metric-label", "3-year risk of FPG-defined incident prediabetes"),
            div(class = "metric-value", textOutput("risk_3y")),
            div(class = "metric-sub", "Predicted cumulative event probability")
          )
        ),
        column(
          width = 4,
          div(
            class = "metric-box",
            div(class = "metric-label", "4-year risk of FPG-defined incident prediabetes"),
            div(class = "metric-value", textOutput("risk_4y")),
            div(class = "metric-sub", "Predicted cumulative event probability")
          )
        ),
        column(
          width = 4,
          div(
            class = "metric-box",
            div(class = "metric-label", "5-year risk of FPG-defined incident prediabetes"),
            div(class = "metric-value", textOutput("risk_5y")),
            div(class = "metric-sub", "Predicted cumulative event probability")
          )
        )
      ),
      
      br(),
      
      div(
        class = "summary-card",
        div(class = "section-title", "Prediction summary"),
        fluidRow(
          column(
            width = 4,
            tags$b("Predicted risk score: "),
            textOutput("risk_score", inline = TRUE)
          ),
          column(
            width = 4,
            tags$b("Risk group: "),
            uiOutput("risk_group_badge")
          ),
          column(
            width = 4,
            tags$b("Main horizon displayed: "),
            span("5 years")
          )
        ),
        div(class = "hr-soft"),
        div(class = "summary-text", htmlOutput("risk_text"))
      ),
      
      br(),
      
      div(
        class = "plot-card",
        div(class = "section-title", "Predicted cumulative risk curve"),
        plotOutput("risk_curve", height = "320px"),
        div(
          class = "plot-caption",
          "Cumulative probability of FPG-defined incident prediabetes during follow-up."
        )
      ),
      
      br(),
      
      div(
        class = "plot-card",
        div(class = "section-title", "Individual SHAP explanation"),
        plotOutput("shap_plot", height = "420px"),
        div(
          class = "plot-caption",
          "Positive SHAP values indicate higher predicted risk; negative SHAP values indicate lower predicted risk."
        )
      )
    )
  )
)


# 5. Server

server <- function(input, output, session) {
  
  current_patient <- reactive({
    data.frame(
      Age = as.numeric(input$Age),
      BMI = as.numeric(input$BMI),
      DBP = as.numeric(input$DBP),
      FPG = as.numeric(input$FPG),
      familyhistroyofdiabetes = as.numeric(input$familyhistroyofdiabetes)
    )
  })
  
  pred_res <- eventReactive(input$predict_btn, {
    new_patient <- current_patient()
    
    res_3 <- predict_surv_risk(new_patient, horizon = 3)
    res_4 <- predict_surv_risk(new_patient, horizon = 4)
    res_5 <- predict_surv_risk(new_patient, horizon = 5)
    
    shap_one <- shapviz(
      xgbmodel_surv,
      X_pred = data.matrix(new_patient[, predictors, drop = FALSE]),
      X = new_patient[, predictors, drop = FALSE]
    )
    
    list(
      new_patient = new_patient,
      res_3 = res_3,
      res_4 = res_4,
      res_5 = res_5,
      shap_one = shap_one
    )
  }, ignoreNULL = FALSE)
  
  output$risk_3y <- renderText({
    fmt_pct(pred_res()$res_3$event_prob)
  })
  
  output$risk_4y <- renderText({
    fmt_pct(pred_res()$res_4$event_prob)
  })
  
  output$risk_5y <- renderText({
    fmt_pct(pred_res()$res_5$event_prob)
  })
  
  output$risk_score <- renderText({
    sprintf("%.3f", pred_res()$res_5$risk_score)
  })
  
  output$risk_group_badge <- renderUI({
    grp <- get_risk_group(pred_res()$res_5$risk_score)
    col <- get_risk_color(grp)
    
    tags$span(
      class = "risk-badge",
      style = paste0("background:", col, ";"),
      grp
    )
  })
  
  output$risk_text <- renderUI({
    risk3 <- pred_res()$res_3$event_prob * 100
    risk4 <- pred_res()$res_4$event_prob * 100
    risk5 <- pred_res()$res_5$event_prob * 100
    grp   <- get_risk_group(pred_res()$res_5$risk_score)
    
    txt <- paste0(
      "According to the time-dependent model, this patient is classified as <b>",
      grp,
      "</b>. The estimated cumulative risk of developing <b>FPG-defined incident prediabetes</b> is <b>",
      round(risk3, 1), "%</b> at 3 years, <b>",
      round(risk4, 1), "%</b> at 4 years, and <b>",
      round(risk5, 1), "%</b> at 5 years. ",
      "These estimates are generated from the combination of <b>Age</b>, <b>Body mass index</b>, <b>Diastolic blood pressure</b>, <b>Fasting plasma glucose</b>, and <b>Family history of diabetes</b>. ",
      "The SHAP plot below explains how each individual variable contributes to the patient-specific risk score."
    )
    
    HTML(txt)
  })
  
  output$risk_curve <- renderPlot({
    build_risk_curve_plot(pred_res()$new_patient, pred_res())
  })
  
  output$shap_plot <- renderPlot({
    build_shap_plot(pred_res()$shap_one)
  })
  
  
  # 下载：风险曲线 PNG
  
  output$download_curve <- downloadHandler(
    filename = function() {
      paste0("FPG_defined_prediabetes_risk_curve_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- build_risk_curve_plot(pred_res()$new_patient, pred_res())
      ggsave(file, plot = p, width = 8, height = 5.2, dpi = 300, bg = "white")
    }
  )
  
  
  # 下载：SHAP 图 PNG
  
  output$download_shap <- downloadHandler(
    filename = function() {
      paste0("FPG_defined_prediabetes_SHAP_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- build_shap_plot(pred_res()$shap_one)
      ggsave(file, plot = p, width = 8, height = 6, dpi = 300, bg = "white")
    }
  )
  
  
  # 下载：预测结果 CSV
  
  output$download_csv <- downloadHandler(
    filename = function() {
      paste0("FPG_defined_prediabetes_prediction_results_", Sys.Date(), ".csv")
    },
    content = function(file) {
      new_patient <- pred_res()$new_patient
      score <- pred_res()$res_5$risk_score
      grp <- get_risk_group(score)
      
      out_df <- data.frame(
        Age = new_patient$Age,
        BMI = new_patient$BMI,
        DBP = new_patient$DBP,
        FPG = new_patient$FPG,
        familyhistroyofdiabetes = new_patient$familyhistroyofdiabetes,
        risk_score = round(score, 6),
        risk_group = grp,
        FPG_defined_prediabetes_risk_3y = round(pred_res()$res_3$event_prob, 6),
        FPG_defined_prediabetes_risk_4y = round(pred_res()$res_4$event_prob, 6),
        FPG_defined_prediabetes_risk_5y = round(pred_res()$res_5$event_prob, 6)
      )
      
      write.csv(out_df, file, row.names = FALSE, fileEncoding = "UTF-8")
    }
  )
  
  
  # 下载：简要报告 HTML
  
  output$download_report <- downloadHandler(
    filename = function() {
      paste0("FPG_defined_prediabetes_prediction_report_", Sys.Date(), ".html")
    },
    content = function(file) {
      html <- build_report_html(pred_res()$new_patient, pred_res())
      writeLines(html, con = file, useBytes = TRUE)
    }
  )
}


# 6. 启动应用

shinyApp(ui = ui, server = server)



