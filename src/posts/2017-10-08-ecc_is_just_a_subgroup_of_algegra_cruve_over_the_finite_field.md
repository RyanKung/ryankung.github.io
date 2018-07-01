# ECC is just a subgroup of Algebra Curve over The Finite Field [Note and Draft]

<canvas id="pdf-canvas"></canvas>

<script>
var url =  "https://github.com/RyanKung/ryankung.github.io/raw/master/src/pdfs/2017-10-08-ecc_is_just_a_subgroup_of_algegra_cruve_over_the_finite_field.pdf" 
var loadingTask = pdfjsLib.getDocument(url);
loadingTask.promise.then(function(pdf) {
  console.log('PDF loaded');
  // Fetch the first page
  var pageNumber = 1;
  pdf.getPage(pageNumber).then(function(page) {
    console.log('Page loaded');
    
    var scale = 1.5;
    var viewport = page.getViewport(scale);

    // Prepare canvas using PDF page dimensions
    var canvas = document.getElementById("pdf-canvas");
    var context = canvas.getContext('2d');
    canvas.height = viewport.height;
    canvas.width = viewport.width;

    // Render PDF page into canvas context
    var renderContext = {
      canvasContext: context,
      viewport: viewport
    };
    var renderTask = page.render(renderContext);
    renderTask.then(function () {
      console.log('Page rendered');
    });
  });
}, function (reason) {
  // PDF loading error
  console.error(reason);
});
</script>

<!-- <embed id="pdfPlayer" src="https://github.com/RyanKung/ryankung.github.io/raw/master/src/pdfs/2017-10-08-ecc_is_just_a_subgroup_of_algegra_cruve_over_the_finite_field.pdf" type="application/pdf" width="100%" height="100%" > -->
