terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "ap-southeast-2"
}

resource "aws_instance" "app_server" {
  ami           = "ami-0f18401e48cde2516"
  instance_type = "t2.micro"
  key_name      = aws_key_pair.my_key_pair.key_name
  security_groups = [aws_security_group.allow_ssh_and_http.name]

  # Read the shell script from the init.sh file and provide it as user_data
  user_data = file("init.sh")

  tags = {
    Name = var.instance_name
  }
}
