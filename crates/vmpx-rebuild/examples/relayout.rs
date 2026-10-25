fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = std::fs::read(&args[1]).expect("read input");
    let (output, report) = vmpx_rebuild::relayout_conventional(&input).expect("relayout");
    std::fs::write(&args[2], &output).expect("write output");
    let validation = vmpx_rebuild::validate_for_loader(&output).expect("loader validation");
    println!(
        "file_size={} headers={:#X} sections={} zerofill={}",
        report.file_size,
        report.size_of_headers,
        report.sections.len(),
        report.zero_fill_sections
    );
    println!("loader: {validation:?}");
}
