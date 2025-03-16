#!/bin/bash

# Set colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==== LLM-Docs Runner Script ====${NC}"
echo

# Check if database exists, initialize if not
if [ ! -f "llm_docs.db" ]; then
  echo -e "${YELLOW}Database not found. Initializing...${NC}"
  python -m llm_docs db init
fi

# Menu
PS3="Select an option: "
options=(
  "Run API server" 
  "Discover packages" 
  "Process a package" 
  "Batch process packages" 
  "List packages" 
  "Show stats" 
  "Reset database" 
  "Run in Docker" 
  "Exit"
)

select opt in "${options[@]}"
do
  case $opt in
    "Run API server")
      echo -e "${GREEN}Starting API server...${NC}"
      python -m llm_docs serve
      break
      ;;
    "Discover packages")
      read -p "How many packages to discover? [100]: " limit
      limit=${limit:-100}
      
      read -p "Process top N packages immediately? [0]: " process
      process=${process:-0}
      
      echo -e "${GREEN}Discovering ${limit} packages...${NC}"
      python -m llm_docs discover --limit $limit --process $process
      break
      ;;
    "Process a package")
      read -p "Package name to process: " package
      
      if [ -z "$package" ]; then
        echo -e "${YELLOW}No package specified.${NC}"
        continue
      fi
      
      echo -e "${GREEN}Processing package ${package}...${NC}"
      python -m llm_docs process $package
      break
      ;;
    "Batch process packages")
      read -p "Enter package names separated by space, or path to file with one package per line: " input
      
      if [ -z "$input" ]; then
        echo -e "${YELLOW}No input provided.${NC}"
        continue
      fi
      
      read -p "Maximum packages to process in parallel [1]: " parallel
      parallel=${parallel:-1}
      
      # Check if input is a file
      if [ -f "$input" ]; then
        echo -e "${GREEN}Processing packages from file ${input} with parallelism ${parallel}...${NC}"
        python -m llm_docs batch --file "$input" --parallel $parallel
      else
        echo -e "${GREEN}Processing packages: ${input} with parallelism ${parallel}...${NC}"
        python -m llm_docs batch $input --parallel $parallel
      fi
      break
      ;;
    "List packages")
      read -p "Filter by status (leave empty for all): " status
      
      if [ -z "$status" ]; then
        echo -e "${GREEN}Listing all packages...${NC}"
        python -m llm_docs list_packages
      else
        echo -e "${GREEN}Listing packages with status ${status}...${NC}"
        python -m llm_docs list_packages --status $status
      fi
      break
      ;;
    "Show stats")
      echo -e "${GREEN}Showing stats...${NC}"
      python -m llm_docs stats
      break
      ;;
    "Reset database")
      echo -e "${YELLOW}WARNING: This will delete all data in the database.${NC}"
      read -p "Are you sure you want to reset the database? [y/N]: " confirm
      
      if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        echo -e "${GREEN}Resetting database...${NC}"
        python -m llm_docs init --reset
      else
        echo "Database reset cancelled."
      fi
      break
      ;;
    "Run in Docker")
      if ! command -v docker-compose &> /dev/null; then
        echo -e "${YELLOW}docker-compose is not installed. Please install Docker and docker-compose.${NC}"
        break
      fi
      
      echo -e "${GREEN}Starting LLM-Docs in Docker...${NC}"
      docker-compose up -d
      echo -e "${GREEN}LLM-Docs API is running at http://localhost:8000${NC}"
      break
      ;;
    "Exit")
      echo "Goodbye!"
      exit 0
      ;;
    *) 
      echo "Invalid option $REPLY"
      ;;
  esac
done
