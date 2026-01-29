# EDA
## movies_metadata.csv
### Bỏ các cột: "poster_path", "homepage", "spoken_languages", "original_title","original_language", "imdb_id"
### Các trường dữ liệu:
#### movies["overview"] = movies["overview"].fillna("")
#### movies["tagline"] = movies["tagline"].fillna("")
#### movies["runtime"] = movies["runtime"].fillna(movies["runtime"].median())
#### movies["budget"] = movies["budget"].fillna(0)
#### movies["revenue"] = movies["revenue"].fillna(0)

## credits.csv
#### Lấy thông tin: cast (diễn viên), director (đạo diễn), writers (biên kịch), producers (nhà sản xuất)


