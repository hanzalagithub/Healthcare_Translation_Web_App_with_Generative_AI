resource "aws_key_pair" "my_key_pair" {
  key_name   = "aws"
  public_key = file("~/.ssh/id_rsa.pub")  # Replace with the path to your public key
}
