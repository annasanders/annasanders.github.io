---
layout: page
title: Pokémon Dashboard
permalink: /r/pkmn_dshbrd
---
### [Live App Here!](https://annasanders.shinyapps.io/Pokemon_dashboard/)

My favorite video game/hobby is Pokémon. Creating a dashboard to search and display all of the available Pokémon 
was a great way to start learning how to program shiny dashboards in R. I am particularly proud of learning how to 
create, add and delete lines from, and reference a newly created table with the app itself. The data was taken from 
[Kaggle](https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420), and the pictures come from [serebii](https://www.serebii.net/index2.shtml). 

### Shiny App Code

```r
library(shiny)
library(tidyverse)

# Load Data
load("pkmn_shinyEnvironment.RData")


# Define UI for application that draws a histogram
pkm_ui_adv2 <- fluidPage(
  
  navbarPage ("Advanced Pokemon Finder",
              #theme = "spacelab", 
              
              tabPanel("Pokemon Finder",
                       
                       column(3, wellPanel(
                         h3("Select Attributes"),
                         
                         actionButton("update", "Update View", width = "100%", class = "btn-primary"),  
                         
                         selectInput("Pkmn_Type_1", 
                                     h4("Pokemon Type 1"), 
                                     choices = c("All",
                                                 unique(pokemon2.1$`Type 1`)),
                                     selected = "",
                                     multiple=TRUE, 
                                     selectize=TRUE),
                         
                         selectInput("Pkmn_Type_2",
                                     h4("Pokemon Type 2"),
                                     choices = c("All",
                                                 unique(pokemon2.1$`Type 2`)),
                                     selected = "",
                                     multiple=TRUE,
                                     selectize=TRUE),
                         
                         sliderInput("Gen",
                                     h4("Generation"),
                                     min = 1,
                                     max = 8,
                                     value = c(1,8)),
                         ticks = FALSE,
                         
                         sliderInput("base_total",
                                     h4("Base Stat Total"),
                                     min = 0,
                                     max = 1125,
                                     value = c(0, 1125),
                                     step = 100),
                         
                         checkboxGroupInput("Legend",
                                            h4("Legendary"),
                                            choices = c(unique(pokemon2.1$`Legendary?`)),
                                            selected = unique(pokemon2.1$`Legendary?`))
                       ),
                       wellPanel(
                         
                         h3("Additional Filters"),
                         
                         selectInput("tbl_ops",
                                     h4("Additional Columns"),
                                     choices = c("All", "Catch Rate", "Base Friendship", 
                                                 "Base XP", "Egg Type 1", "Egg Type 2", "Percent Male"),
                                     selected = "",
                                     multiple=TRUE,
                                     selectize=TRUE),        
                         
                         sliderInput("atk",
                                     h4("Attack"),
                                     min = 0,
                                     max = 250,
                                     value = c(0, 250)),
                         
                         sliderInput("def",
                                     h4("Defense"),
                                     min = 0,
                                     max = 250,
                                     value = c(0, 250)),
                         
                         sliderInput("sp_atk",
                                     h4("Special Attack"),
                                     min = 0,
                                     max = 250,
                                     value = c(0, 250)),
                         
                         sliderInput("sp_def",
                                     h4("Special Defense"),
                                     min = 0,
                                     max = 250,
                                     value = c(0, 250)),
                         
                         sliderInput("spd",
                                     h4("Speed"),
                                     min = 0,
                                     max = 250,
                                     value = c(0, 250)) 
                         
                       )),
                       fluidRow(
                         column(8,
                                
                                # h4("Type 1:"),
                                # textOutput("selected_type_1"),
                                # 
                                # h4("Type 2:"),
                                # textOutput("selected_type_2"),
                                # 
                                # h4("Testing"),
                                # textOutput("testing"),
                                
                                DT::dataTableOutput("found_pkmn", width = "110%"), #table width with scrollable table (in datatable options),
                                
                                actionButton("add_pkmn", "Add Pokemon to List", class = "btn-primary", width = "50%"),
                                
                                
                                uiOutput("pkmn_img1") 
                         ))),
              
              tabPanel("My Team",
                       fluidRow(
                         column(3,
                                actionButton("del_pkmn", "Remove Pokemon", class = "btn-primary", width = "100%"))
                         
                       ),
                       br(),
                       fluidRow(
                         column(10,
                                DT::dataTableOutput("pkmn_list", width = "100%"),
                                downloadButton("dwn_list", "Download List (.csv)", class = "btn-primary", width = "60%")),
                         h4("Strong Against: "), textOutput("strong_against"),
                         br(),
                         h4("Weak Against: "), textOutput("weak_against"),
                         uiOutput("pkmn_img2")
                       )    
              )
              
  )
)


pkm_server_adv2 <- function(input,output){
  
  # output$selected_type_1 <- renderText({paste(input$Pkmn_Type_1)})
  # 
  # output$selected_type_2 <- renderText({paste(input$Pkmn_Type_2)})
  # 
  # output$testing <- renderText({paste(pokemon2.2[(update_pkmn()$`index`[input$found_pkmn_rows_selected]+1),])})
  ## output$testing <- renderText({paste("All" %in% input$Pkmn_Type_1)})
  #output$testing <- renderText(paste(pokemon_list_r$x))
  
  output$found_pkmn <- DT::renderDataTable(DT::datatable(
    update_pkmn(), options = list(lengthMenu = c(10, 25, 50), pageLength = 10, autoWidth = TRUE, scrollX = TRUE, columnDefs = list(list(visible=FALSE, targets=c(0,1)))), selection = "single"
  ))
  
  # First Tab 
  update_pkmn <- eventReactive(input$update,{
    pokemon2.1_s <- pokemon2.1
    
    # Main Bar
    if ((is.null(isolate(input$Pkmn_Type_1)) == TRUE) & (is.null(isolate(input$Pkmn_Type_2)) == TRUE))
    {}
    else if ((isolate(input$Pkmn_Type_1) %in% isolate(input$Pkmn_Type_2)) & 
             ("All" %in% input$Pkmn_Type_1 == FALSE))
    {pokemon2.1_s <- rbind(pokemon2.1[pokemon2.1$`Type 1` %in% input$Pkmn_Type_1,],
                           pokemon2.1[pokemon2.1$`Type 2` %in% input$Pkmn_Type_2,])}
    else {
      if ((is.null(isolate(input$Pkmn_Type_1)) == TRUE) || ("All" %in% input$Pkmn_Type_1 == TRUE))
      {} 
      else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Type 1` %in% input$Pkmn_Type_1,]}
      
      if ((is.null(isolate(input$Pkmn_Type_2)) == TRUE) || ("All" %in% input$Pkmn_Type_2 == TRUE))
      {}
      else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Type 2` %in% input$Pkmn_Type_2,]}
    }
    
    if ((is.null(isolate(input$Gen)) == TRUE) || (input$Gen[1] == 1 && input$Gen[2] == 8))
    {}
    else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$Gen >= input$Gen[1],]
    pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$Gen <= input$Gen[2],]}    
    
    if (is.null(isolate(input$Legend)) == TRUE)
    {}
    else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Legendary?` %in% input$Legend,]}
    
    if ((is.null(isolate(input$base_total)) == TRUE) || (input$base_total[1] == 0 & input$base_total[2] == 1125))
    {}
    else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Base Stat Total` >= input$base_total[1],]
    pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Base Stat Total` <= input$base_total[2],]}
    
    # Secondary Bar
    if ((is.null(isolate(input$atk)) == TRUE) || (input$atk[1] == 0 & input$atk[2] == 250))
    {}
    else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Attack` >= input$atk[1],]
    pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Attack` <= input$atk[2],]}
    
    if ((is.null(isolate(input$def)) == TRUE) || (input$def[1] == 0 & input$def[2] == 250))
    {}
    else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Defense` >= input$def[1],]
    pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Defense` <= input$def[2],]}   
    
    if ((is.null(isolate(input$sp_atk)) == TRUE) || (input$sp_atk[1] == 0 & input$sp_atk[2] == 250))
    {}
    else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Special Attack` >= input$sp_atk[1],]
    pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Special Attack` <= input$sp_atk[2],]}   
    
    if ((is.null(isolate(input$sp_def)) == TRUE) || (input$sp_def[1] == 0 & input$sp_def[2] == 250))
    {}
    else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Special Defense` >= input$sp_def[1],]
    pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Special Defense` <= input$sp_def[2],]}
    
    if ((is.null(isolate(input$spd)) == TRUE) || (input$spd[1] == 0 & input$spd[2] == 250))
    {}
    else {pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Speed` >= input$spd[1],]
    pokemon2.1_s <- pokemon2.1_s[pokemon2.1_s$`Speed` <= input$spd[2],]}   
    
    # Table Options  
    if ((is.null(isolate(input$tbl_ops)) == TRUE))
    {pokemon2.1_s <- subset(pokemon2.1_s, select = -c(`Catch Rate`, `Base Friendship`, 
                                                      `Base XP`, `Egg Group 1`, `Egg Group 2`, `Percent Male`))}
    else if (isolate(input$tbl_ops) %in% "All" == TRUE) 
    {}
    else {pokemon2.1_s <- subset(pokemon2.1_s, select = -(c(`Catch Rate`, `Base Friendship`, 
                                                            `Base XP`, `Egg Group 1`, `Egg Group 2`, `Percent Male`)[!(c("Catch Rate", "Base Friendship", "Base XP", "Egg Group 1", "Egg Group 2", "Percent Male") %in% input$tbl_ops)]))}   
    
    #pokemon2.1_s <- slice_min(pokemon2.1_s, n = 50, order_by = pokemon2.1_s$`Dex Number`, with_ties = F) #Not needed as the datatable does a good job collapsing
    
    pokemon2.1_s})
  
  # Image Generation
  output$pkmn_img1 <- renderUI(
    if (is.null(input$found_pkmn_rows_selected) == TRUE)
    {}
    else
    {tags$img(src = paste("https://www.serebii.net/pokemon/art/",{str_pad(update_pkmn()$`Dex Number`[input$found_pkmn_rows_selected], 3, pad = "0")},".png", sep = ""))})
  
  # Second Tab    
  pokemon_list_r <- reactiveValues(x = pokemon_list)
  proxy <- DT::dataTableProxy("pkmn_list")
  observe(DT::replaceData(proxy, pokemon_list_r$x))
  
  # Download List Table
  # output$testing <- renderText({paste(dplyr::bind_rows(pokemon_list_r$x))})
  
  output$dwn_list <- downloadHandler(
    filename = function() {
      paste("my pokemon", ".csv", sep = "")},
    content = function(file) {
      write.csv({dplyr::bind_rows(pokemon_list_r$x)}, file, row.names = TRUE)}
  )
  
  # Adding to New Table
  observeEvent(input$add_pkmn, {
    if (is.null(input$found_pkmn_rows_selected) == TRUE)
    {}
    else 
    {pokemon_list_r$x <- pokemon_list_r$x %>%
      bind_rows(
        {pokemon2.2[(update_pkmn()$index[input$found_pkmn_rows_selected]+1),]})}
  }
  )
  
  observeEvent(input$del_pkmn, {
    if (is.null(input$pkmn_list_rows_selected) == TRUE)
    {}
    else
    {pokemon_list_r$x <- pokemon_list_r$x[-c(input$pkmn_list_rows_selected),]}
  }
  )
  
  # Table Output
  output$pkmn_list <- DT::renderDataTable(pokemon_list, options = list(autoWidth = TRUE, scrollX = TRUE, columnDefs = list(list(visible=FALSE, targets=c(0,1, 4:5, 8:9, 14, 25:51)))),
                                          selection = "single"
  )
  
  
  # Image Generation
  output$pkmn_img2 <- renderUI(
    if (is.null(input$pkmn_list_rows_selected) == TRUE)
    {}
    else
    {tags$img(src = paste("https://www.serebii.net/pokemon/art/",{str_pad(pokemon_list_r$x[input$pkmn_list_rows_selected,2], 3, pad = "0")},".png", sep = ""))})
  
  
  # Type Advantages
  # output$testing <- renderText({paste((0.5 %in% pokemon_list_r$x[,50] == TRUE) || (0.25 %in% pokemon_list_r$x[,50] == TRUE) || (0 %in% pokemon_list_r$x[,50] == TRUE))})
  
  output$strong_against <- renderText({
    types(1)
  })
  
  output$weak_against <- renderText({
    types(0)
  })    
  
  types <- function(type_p){
    type_dis <- c()
    type_adv <- c()
    
    if ((1 %in% pokemon_list_r$x[,51] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,51] == TRUE) || (0.25 %in% pokemon_list_r$x[,51] == TRUE) || (0 %in% pokemon_list_r$x[,51] == TRUE)) {type_adv <- c(type_adv, "Fairy") }
    else {type_dis <- c(type_dis, "Fairy") }
    
    if ((1 %in% pokemon_list_r$x[,50] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,50] == TRUE) || (0.25 %in% pokemon_list_r$x[,50] == TRUE) || (0 %in% pokemon_list_r$x[,50] == TRUE)) {type_adv <- c(type_adv, "Steel") }
    else {type_dis <- c(type_dis, "Steel") }
    
    if ((1 %in% pokemon_list_r$x[,49] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,49] == TRUE) || (0.25 %in% pokemon_list_r$x[,49] == TRUE) || (0 %in% pokemon_list_r$x[,49] == TRUE)) {type_adv <- c(type_adv, "Dark") }
    else {type_dis <- c(type_dis, "Dark") } 
    
    if ((1 %in% pokemon_list_r$x[,48] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,48] == TRUE) || (0.25 %in% pokemon_list_r$x[,48] == TRUE) || (0 %in% pokemon_list_r$x[,48] == TRUE)) {type_adv <- c(type_adv, "Dragon") }
    else {type_dis <- c(type_dis, "Dragon") }    
    
    if ((1 %in% pokemon_list_r$x[,47] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,47] == TRUE) || (0.25 %in% pokemon_list_r$x[,47] == TRUE) || (0 %in% pokemon_list_r$x[,47] == TRUE)) {type_adv <- c(type_adv, "Ghost") }
    else {type_dis <- c(type_dis, "Ghost") } 
    
    if ((1 %in% pokemon_list_r$x[,46] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,46] == TRUE) || (0.25 %in% pokemon_list_r$x[,46] == TRUE) || (0 %in% pokemon_list_r$x[,46] == TRUE)) {type_adv <- c(type_adv, "Rock") }
    else {type_dis <- c(type_dis, "Rock") }
    
    if ((1 %in% pokemon_list_r$x[,45] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,45] == TRUE) || (0.25 %in% pokemon_list_r$x[,45] == TRUE) || (0 %in% pokemon_list_r$x[,45] == TRUE)) {type_adv <- c(type_adv, "Bug") }
    else {type_dis <- c(type_dis, "Bug") }     
    
    if ((1 %in% pokemon_list_r$x[,44] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,44] == TRUE) || (0.25 %in% pokemon_list_r$x[,44] == TRUE) || (0 %in% pokemon_list_r$x[,44] == TRUE)) {type_adv <- c(type_adv, "Psychic") }
    else {type_dis <- c(type_dis, "Psychic") }     
    
    if ((1 %in% pokemon_list_r$x[,43] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,43] == TRUE) || (0.25 %in% pokemon_list_r$x[,43] == TRUE) || (0 %in% pokemon_list_r$x[,43] == TRUE)) {type_adv <- c(type_adv, "Flying") }
    else {type_dis <- c(type_dis, "Flying") }
    
    if ((1 %in% pokemon_list_r$x[,42] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,42] == TRUE) || (0.25 %in% pokemon_list_r$x[,42] == TRUE) || (0 %in% pokemon_list_r$x[,42] == TRUE)) {type_adv <- c(type_adv, "Ground") }
    else {type_dis <- c(type_dis, "Ground") }
    
    if ((1 %in% pokemon_list_r$x[,41] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,41] == TRUE) || (0.25 %in% pokemon_list_r$x[,41] == TRUE) || (0 %in% pokemon_list_r$x[,41] == TRUE)) {type_adv <- c(type_adv, "Poison") }
    else {type_dis <- c(type_dis, "Poison") }
    
    if ((1 %in% pokemon_list_r$x[,40] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,40] == TRUE) || (0.25 %in% pokemon_list_r$x[,40] == TRUE) || (0 %in% pokemon_list_r$x[,40] == TRUE)) {type_adv <- c(type_adv, "Fighting") }
    else {type_dis <- c(type_dis, "Fighting") }  
    
    if ((1 %in% pokemon_list_r$x[,39] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,39] == TRUE) || (0.25 %in% pokemon_list_r$x[,39] == TRUE) || (0 %in% pokemon_list_r$x[,39] == TRUE)) {type_adv <- c(type_adv, "Ice") }
    else {type_dis <- c(type_dis, "Ice") }   
    
    if ((1 %in% pokemon_list_r$x[,38] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,38] == TRUE) || (0.25 %in% pokemon_list_r$x[,38] == TRUE) || (0 %in% pokemon_list_r$x[,38] == TRUE)) {type_adv <- c(type_adv, "Grass") }
    else {type_dis <- c(type_dis, "Grass") }  
    
    if ((1 %in% pokemon_list_r$x[,37] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,37] == TRUE) || (0.25 %in% pokemon_list_r$x[,37] == TRUE) || (0 %in% pokemon_list_r$x[,37] == TRUE)) {type_adv <- c(type_adv, "Electric") }
    else {type_dis <- c(type_dis, "Electric") } 
    
    if ((1 %in% pokemon_list_r$x[,36] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,36] == TRUE) || (0.25 %in% pokemon_list_r$x[,36] == TRUE) || (0 %in% pokemon_list_r$x[,36] == TRUE)) {type_adv <- c(type_adv, "Water") }
    else {type_dis <- c(type_dis, "Water") } 
    
    if ((1 %in% pokemon_list_r$x[,35] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,35] == TRUE) || (0.25 %in% pokemon_list_r$x[,35] == TRUE) || (0 %in% pokemon_list_r$x[,35] == TRUE)) {type_adv <- c(type_adv, "Fire") }
    else {type_dis <- c(type_dis, "Fire") }
    
    if ((1 %in% pokemon_list_r$x[,34] == TRUE)) {}
    else if ((0.5 %in% pokemon_list_r$x[,34] == TRUE) || (0.25 %in% pokemon_list_r$x[,34] == TRUE) || (0 %in% pokemon_list_r$x[,34] == TRUE)) {type_adv <- c(type_adv, "Normal") }
    else {type_dis <- c(type_dis, "Normal") } 
    
    if (type_p == 1) {paste(type_adv, collapse=", ")}
    else {paste(type_dis, collapse=", ")}
  }
}

shinyApp(ui = pkm_ui_adv2, server = pkm_server_adv2)
```

