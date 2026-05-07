terraform {
  required_version = ">= 1.4"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
  # Configuración para LocalStack
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  endpoints {
    s3  = "http://localhost:4566"
    ecr = "http://localhost:4566"
  }
  default_tags {
    tags = {
      Proyecto   = "creditml"
      Asignatura = "20GIAR"
      Entorno    = "local"
    }
  }
}

# ── S3: bucket para datos raw y procesados ──────────────────────────────────

resource "aws_s3_bucket" "datos" {
  bucket = "creditml-data-local"
}

resource "aws_s3_bucket_versioning" "datos" {
  bucket = aws_s3_bucket.datos.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datos" {
  bucket = aws_s3_bucket.datos.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ── ECR: repositorio simulado para imágenes Docker ──────────────────────────

resource "aws_ecr_repository" "api" {
  name                 = "creditml-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false  # Deshabilitado para LocalStack
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Mantener solo las últimas 5 imágenes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}