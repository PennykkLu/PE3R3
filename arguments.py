def predefined_args(parser):
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--inf_batch_size', default=1000, type=int)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--gpu_id', default=2, type=int)
    parser.add_argument('--dataset', default='../dataset/Amazon_sum/Instant_Video_5.csv', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--lr', default=0.007, type=float)
    parser.add_argument('--patience', default=20, type=int)

    # para for residual ensemble
    parser.add_argument('--d', type=int, help="codebook num", default=3)
    parser.add_argument('--codename', type=str, default="Instant")
    parser.add_argument('--cdb_path', type=str, default="./codebook/centroids_embedding_")
    parser.add_argument('--sim_mode', type=str, choices=["dot", "cos"], default="dot")

    return parser
