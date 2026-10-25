fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = std::fs::read(&args[1]).unwrap();
    let (out, report) = vmpx_rebuild::relayout_conventional(&input).unwrap();
    std::fs::write(&args[2], &out).unwrap();
    println!(
        "file_size={} headers={:#X} sections={} zerofill={}",
        report.file_size,
        report.size_of_headers,
        report.sections.len(),
        report.zero_fill_sections
    );
}
